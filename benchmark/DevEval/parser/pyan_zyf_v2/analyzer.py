#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The AST visitor."""

import ast
import logging
import symtable
from typing import List, Union
from func_timeout import func_set_timeout
import func_timeout

from .anutils import (
    ExecuteInInnerScope,
    Scope,
    UnresolvedSuperCallError,
    format_alias,
    get_ast_node_name,
    get_module_name,
    resolve_method_resolution_order,
    sanitize_exprs,
    tail,
)
from .node import Flavor, Node

import traceback


# TODO: add Cython support (strip type annotations in a preprocess step, then treat as Python)
# TODO: built-in functions (range(), enumerate(), zip(), iter(), ...):
#       add to a special scope "built-in" in analyze_scopes() (or ignore altogether)
# TODO: support Node-ifying ListComp et al, List, Tuple
# TODO: make the analyzer smarter (see individual TODOs below)

# qika TODO: 2. 由于每个变量都只记录关系链中的最后一个节点作为其状态，则无法将中间节点加入调用关系图中
# 例如，a = A(); b=B()中有个b1=a，则调用图中只有a，而没有b.b1
# 或者 a引用了一个import A的东西，而这个被import A的东西是在别的地方定义的A，那么调用图中只有定义的位置，而无法将 import这个信息加入到调用图中
# 暂定处理方法：修改Node类，使其保存三个属性，第一个是这个Node定义处对应的子节点，的不将变量关系链中的最终类型赋值给变量，而是将关系链中下一个节点作为变量的新的“类型”参数保存
# 对节点新建Use关系的同时递归的对其类型也建立Use关系，直到遇到没有类型的节点为止

# qika：修改self.last_value 记录actual_path（String类型），而非节点。actual_path可以不对应任何真实或虚拟节点

# TODO qika:3 特殊处理 Optional[xxx] 类型，将其视为xxx类型


# Note the use of the term "node" for two different concepts:
#
#  - AST nodes (the "node" argument of CallGraphVisitor.visit_*())
#
#  - The Node class that mainly stores auxiliary information about AST nodes,
#    for the purposes of generating the call graph.
#
#    Namespaces also get a Node (with no associated AST node).

# These tables were useful for porting the visitor to Python 3:
#
# https://docs.python.org/2/library/compiler.html#module-compiler.ast
# https://docs.python.org/3/library/ast.html#abstract-grammar
#


class CallGraphVisitor(ast.NodeVisitor):
    """A visitor that can be walked over a Python AST, and will derive
    information about the objects in the AST and how they use each other.

    A single CallGraphVisitor object can be run over several ASTs (from a
    set of source files).  The resulting information is the aggregate from
    all files.  This way use information between objects in different files
    can be gathered."""

    def __init__(self, filenames, root: str = None, logger=None):
        self.logger = logger or logging.getLogger(__name__)

        # full module names for all given files
        self.module_to_filename = {}  # inverse mapping for recording which file each AST node came from
        for filename in filenames:
            mod_name = get_module_name(filename)
            self.module_to_filename[mod_name] = filename
        self.filenames = filenames
        self.root = root
        
        self.functional_info = {} # 保存类、函数、（Block）级别对应的功能注释等信息，默认为body中的第一个注释

        # data gathered from analysis
        self.defines_edges = {}
        self.uses_edges = {}
        self.import_uses_edges = {}  # 对于通过import引入的名称，记录其使用关系
        self.virtual_uses_edges = {} # 对于找不到具体定义位置的名称，将其作为虚拟节点，用于记录其使用关系
        self.relevant_edges = {}
        self.nodes = {}  # Node name: list of Node objects (in possibly different namespaces)
        self.scopes = {}  # fully qualified name of namespace: Scope object

        self.class_base_ast_nodes = {}  # pass 1: class Node: list of AST nodes
        self.class_base_nodes = {}  # pass 2: class Node: list of Node objects (local bases, no recursion)
        self.mro = {}  # pass 2: class Node: list of Node objects in Python's MRO order

        # current context for analysis
        self.module_name = None
        self.filename = None
        self.name_stack = []  # for building namespace name, node naming
        self.scope_stack = []  # the Scope objects currently in scope
        self.class_stack = []  # Nodes for class definitions currently in scope
        self.context_stack = []  # for detecting which FunctionDefs are methods
        
        # qika: 用于记录之前处理过的代码信息，用于后续处理（赋值或调用）
        # last_value: 之前处理过的代码对应的虚拟节点
        # last_type: 之前处理过的代码对应的虚拟节点的实际类型
        # 如无法确定，则为None
        self.last_value = None
        self.last_type = None
        
        
        # 记录当前是第几次process_one
        self.pas_time = 0

        # 调试专用
        self.qika_effective_principals = 0

        # Analyze.
        self.process()

    def process(self):
        """Analyze the set of files, twice so that any forward-references are picked up."""
        for pas in range(2):
            self.pas_time = pas
            from tqdm import tqdm
            pbar = tqdm(self.filenames)
            for filename in pbar:
                pbar.set_description('analyze '+filename)
                self.logger.info("========== pass %d, file '%s' ==========" % (pas + 1, filename))
                try:
                    self.process_one(filename)
                except func_timeout.exceptions.FunctionTimedOut:
                    print('time out '+filename)
                    self.name_stack = []
                    self.class_stack = []
                    self.scope_stack = []
                    self.context_stack = []
                except Exception as e:
                    print(traceback.format_exc())
                    print("error "+filename)
                    self.name_stack = []
                    self.class_stack = []
                    self.scope_stack = []
                    self.context_stack = []
            if pas == 0:
                self.resolve_base_classes()  # must be done only after all files seen
        self.postprocess()

    # @func_set_timeout(30)
    def process_one(self, filename):
        """Analyze the specified Python source file."""
        if filename not in self.filenames:
            raise ValueError(
                "Filename '%s' has not been preprocessed (was not given to __init__, which got %s)"
                % (filename, self.filenames)
            )
        with open(filename, "rt", encoding="utf-8") as f:
            content = f.read()
        self.filename = filename
        self.module_name = get_module_name(filename, root=self.root)
        self.analyze_scopes(content, filename)  # add to the currently known scopes
        self.visit(ast.parse(content, filename))
        self.module_name = None
        self.filename = None
        
    # @func_set_timeout(30)
    def add_process_one(self, filename, content):
        self.filename = filename
        self.module_name = get_module_name(filename, root=self.root)
        self.analyze_scopes(content, filename)  # add to the currently known scopes
        self.visit(ast.parse(content, filename))
        self.module_name = None
        self.filename = None

    def resolve_base_classes(self):
        """Resolve base classes from AST nodes to Nodes.

        Run this between pass 1 and pass 2 to pick up inherited methods.
        Currently, this can parse ast.Names and ast.Attributes as bases.
        """
        self.logger.debug("Resolving base classes")
        assert len(self.scope_stack) == 0  # only allowed between passes
        for node in self.class_base_ast_nodes:  # Node: list of AST nodes
            self.class_base_nodes[node] = []
            for ast_node in self.class_base_ast_nodes[node]:
                # perform the lookup in the scope enclosing the class definition
                self.scope_stack.append(self.scopes[node.namespace])

                if isinstance(ast_node, ast.Name):
                    baseclass_node, baseclass_sc = self.get_value(ast_node.id)
                elif isinstance(ast_node, ast.Attribute):
                    _, baseclass_node = self.get_attribute(ast_node)  # don't care about obj, just grab attr
                else:  # give up
                    baseclass_node = None

                self.scope_stack.pop()

                if isinstance(baseclass_node, Node) and baseclass_node.namespace is not None:
                    self.class_base_nodes[node].append(baseclass_node)

        self.logger.debug("All base classes (non-recursive, local level only): %s" % self.class_base_nodes)

        self.logger.debug("Resolving method resolution order (MRO) for all analyzed classes")
        self.mro = resolve_method_resolution_order(self.class_base_nodes, self.logger)
        self.logger.debug("Method resolution order (MRO) for all analyzed classes: %s" % self.mro)

    def postprocess(self):
        """Finalize the analysis."""

        # Compared to the original Pyan, the ordering of expand_unknowns() and
        # contract_nonexistents() has been switched.
        #
        # It seems the original idea was to first convert any unresolved, but
        # specific, references to the form *.name, and then expand those to see
        # if they match anything else. However, this approach has the potential
        # to produce a lot of spurious uses edges (for unrelated functions with
        # a name that happens to match).
        #
        # Now that the analyzer is (very slightly) smarter about resolving
        # attributes and imports, we do it the other way around: we only expand
        # those references that could not be resolved to any known name, and
        # then remove any references pointing outside the analyzed file set.

        # self.expand_unknowns()
        # self.resolve_imports()
        # self.contract_nonexistents()
        # self.cull_inherited()
        self.collapse_inner()

    ###########################################################################
    # visitor methods

    # In visit_*(), the "node" argument refers to an AST node.

    # Python docs:
    # https://docs.python.org/3/library/ast.html#abstract-grammar

    def resolve_imports(self):
        """
        resolve relative imports and remap nodes
        """
        # first find all imports and map to themselves. we will then remap those that are currently pointing
        # to duplicates or into the void
        imports_to_resolve = {n for items in self.nodes.values() for n in items if n.flavor == Flavor.IMPORTEDITEM}
        # map real definitions
        import_mapping = {}
        while len(imports_to_resolve) > 0:
            from_node = imports_to_resolve.pop()
            if from_node in import_mapping:
                continue
            to_uses = self.uses_edges.get(from_node, set([from_node]))
            assert len(to_uses) == 1
            to_node = to_uses.pop()  # resolve alias
            # resolve namespace and get module
            if to_node.namespace == "":
                module_node = to_node
            else:
                assert from_node.name == to_node.name
                module_node = self.get_node("", to_node.namespace)
            module_uses = self.uses_edges.get(module_node)
            if module_uses is not None:
                # check if in module item exists and if yes, map to it
                for candidate_to_node in module_uses:
                    if candidate_to_node.name == from_node.name:
                        to_node = candidate_to_node
                        import_mapping[from_node] = to_node
                        if to_node.flavor == Flavor.IMPORTEDITEM and from_node is not to_node:  # avoid self-recursion
                            imports_to_resolve.add(to_node)
                        break

        # set previously undefined nodes to defined
        # go through undefined attributes
        attribute_import_mapping = {}
        for nodes in self.nodes.values():
            for node in nodes:
                if not node.defined and node.flavor == Flavor.ATTRIBUTE:
                    # try to resolve namespace and find imported item mapping
                    for from_node, to_node in import_mapping.items():
                        if (
                            f"{from_node.namespace}.{from_node.name}" == node.namespace
                            and from_node.flavor == Flavor.IMPORTEDITEM
                        ):
                            if to_node not in self.defines_edges:
                                self.defines_edges[to_node] = set()
                            # use define edges as potential candidates
                            for candidate_to_node in self.defines_edges[to_node]:  #
                                if candidate_to_node.name == node.name:
                                    attribute_import_mapping[node] = candidate_to_node
                                    break
        import_mapping.update(attribute_import_mapping)

        # remap nodes based on import mapping
        self.nodes = {name: [import_mapping.get(n, n) for n in items] for name, items in self.nodes.items()}
        self.uses_edges = {
            import_mapping.get(from_node, from_node): {import_mapping.get(to_node, to_node) for to_node in to_nodes}
            for from_node, to_nodes in self.uses_edges.items()
            if len(to_nodes) > 0
        }
        self.defines_edges = {
            import_mapping.get(from_node, from_node): {import_mapping.get(to_node, to_node) for to_node in to_nodes}
            for from_node, to_nodes in self.defines_edges.items()
            if len(to_nodes) > 0
        }

    def filter(self, node: Union[None, Node] = None, namespace: Union[str, None] = None, max_iter: int = 1000):
        """
        filter callgraph nodes that related to `node` or are in `namespace`

        Args:
            node: pyan node for which related nodes should be found, if none, filter only for namespace
            namespace: namespace to search in (name of top level module),
                if None, determines namespace from `node`
            max_iter: maximum number of iterations and nodes to iterate

        Returns:
            self
        """
        # filter the nodes to avoid cluttering the callgraph with irrelevant information
        filtered_nodes = self.get_related_nodes(node, namespace=namespace, max_iter=max_iter)

        self.nodes = {name: [node for node in nodes if node in filtered_nodes] for name, nodes in self.nodes.items()}
        self.uses_edges = {
            node: {n for n in nodes if n in filtered_nodes}
            for node, nodes in self.uses_edges.items()
            if node in filtered_nodes
        }
        self.defines_edges = {
            node: {n for n in nodes if n in filtered_nodes}
            for node, nodes in self.defines_edges.items()
            if node in filtered_nodes
        }
        return self

    def get_related_nodes(
        self, node: Union[None, Node] = None, namespace: Union[str, None] = None, max_iter: int = 1000):
        """
        get nodes that related to `node` or are in `namespace`

        Args:
            node: pyan node for which related nodes should be found, if none, filter only for namespace
            namespace: namespace to search in (name of top level module),
                if None, determines namespace from `node`
            max_iter: maximum number of iterations and nodes to iterate

        Returns:
            set: set of nodes related to `node` including `node` itself
        """
        # check if searching through all nodes is necessary
        if node is None:
            queue = []
            if namespace is None:
                new_nodes = {n for items in self.nodes.values() for n in items}
            else:
                new_nodes = {
                    n
                    for items in self.nodes.values()
                    for n in items
                    if n.namespace is not None and namespace in n.namespace
                }

        else:
            new_nodes = set()
            if namespace is None:
                namespace = node.namespace.strip(".").split(".", 1)[0]
            queue = [node]

        # use queue system to search through nodes
        # essentially add a node to the queue and then search all connected nodes which are in turn added to the queue
        # until the queue itself is empty or the maximum limit of max_iter searches have been hit
        i = max_iter
        while len(queue) > 0:
            item = queue.pop()
            if item not in new_nodes:
                new_nodes.add(item)
                i -= 1
                if i < 0:
                    break
                queue.extend(
                    [
                        n
                        for n in self.uses_edges.get(item, [])
                        if n in self.uses_edges and n not in new_nodes and namespace in n.namespace
                    ]
                )
                queue.extend(
                    [
                        n
                        for n in self.defines_edges.get(item, [])
                        if n in self.defines_edges and n not in new_nodes and namespace in n.namespace
                    ]
                )

        return new_nodes

    def visit_Module(self, node):
        self.logger.debug("Module %s, %s" % (self.module_name, self.filename))

        # Modules live in the top-level namespace, ''.
        module_ns, module_name = self.split(self.module_name)
        module_node = self.get_node(module_ns, module_name, node, flavor=Flavor.MODULE)
        module_node_2 = self.get_node("", self.module_name, node, flavor=Flavor.MODULE)

        # Module的定义节点的类型路径是它自己, 定义路径也是它自己 
        module_node.set_type(module_node.get_name())
        module_node.set_defined_path(module_node.get_name())
        self.associate_node(module_node, node, filename=self.filename)

        ns = self.module_name
        self.name_stack.append(ns)
        self.scope_stack.append(self.scopes[ns])
        self.context_stack.append("Module %s" % (ns))
        self.generic_visit(node)  # visit the **children** of node
        self.context_stack.pop()
        self.scope_stack.pop()
        self.name_stack.pop()
        self.last_value = None

        if self.add_defines_edge(module_node, None):
            self.logger.info("Def Module %s" % node)
        
        if module_node.get_name() not in self.uses_edges:
            self.uses_edges[module_node.get_name()] = set()
        if module_node.get_name() not in self.virtual_uses_edges:
            self.virtual_uses_edges[module_node.get_name()] = {}
    
    def visit_ClassDef(self, node):
        self.logger.debug("ClassDef %s, %s:%s" % (node.name, self.filename, node.lineno))
        
        if node.name == "CallbackAuthenticationPolicy":
            self.qika_effective_principals = 1

        from_node = self.get_node_of_current_namespace()
        ns = from_node.get_name()
        class_node = self.get_node(ns, node.name, node, flavor=Flavor.CLASS)
        # Class的定义节点的类型路径是它自己, 定义路径也是它自己
        class_node.set_type(class_node.get_name())
        class_node.set_defined_path(class_node.get_name())
        if self.add_defines_edge(from_node, class_node):
            self.logger.info("Def from %s to Class %s" % (from_node, class_node))

        if class_node.get_name() not in self.uses_edges:
            self.uses_edges[class_node.get_name()] = set()
        if class_node.get_name() not in self.virtual_uses_edges:
            self.virtual_uses_edges[class_node.get_name()] = {}
        
        # The graph Node may have been created earlier by a FromImport,
        # in which case its AST node points to the site of the import.
        #
        # Change the AST node association of the relevant graph Node
        # to this AST node (the definition site) to get the correct
        # source line number information in annotated output.
        #
        self.associate_node(class_node, node, self.filename)

        # Bind the name specified by the AST node to the graph Node
        # in the current scope.
        #
        self.set_value(node.name, new_value=class_node)

        self.class_stack.append(class_node)
        self.name_stack.append(node.name)
        inner_ns = self.get_node_of_current_namespace().get_name()
        self.scope_stack.append(self.scopes[inner_ns])
        self.context_stack.append("ClassDef %s" % (node.name))

        self.class_base_ast_nodes[class_node] = []
        for b in node.bases:
            # gather info for resolution of inherited attributes in pass 2 (see get_attribute())
            self.class_base_ast_nodes[class_node].append(b)
            # mark uses from a derived class to its bases (via names appearing in a load context).
            self.visit(b)


        if class_node.get_name() == 'xmnlp.module.Module':
            qika = 1
        for stmt in node.body:
            self.visit(stmt)

        self.context_stack.pop()
        self.scope_stack.pop()
        self.name_stack.pop()
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        self.logger.debug("FunctionDef %s, %s:%s" % (node.name, self.filename, node.lineno))

        
        
        # To begin with:
        #
        # - Analyze decorators. They belong to the surrounding scope,
        #   so we must analyze them before entering the function scope.
        #
        # - Determine whether this definition is for a function, an (instance)
        #   method, a static method or a class method.
        #
        # - Grab the name representing "self", if this is either an instance
        #   method or a class method. (For a class method, it represents cls,
        #   but Pyan only cares about types, not instances.)
        #
        self_name, flavor = self.analyze_functiondef(node)

        # Now we can create the Node.
        #
        from_node = self.get_node_of_current_namespace()
        ns = from_node.get_name()
        func_node = self.get_node(ns, node.name, node, flavor=flavor)
        func_node.set_type(func_node.get_name())
        func_node.set_value(func_node)
        if self.add_defines_edge(from_node, func_node):
            self.logger.info("Def from %s to Function %s" % (from_node, func_node))
        
        self.uses_edges[func_node.get_name()] = set()
        
        
        if func_node.get_name() not in self.uses_edges:
            self.uses_edges[func_node.get_name()] = set()
        if func_node.get_name() not in self.virtual_uses_edges:
            self.virtual_uses_edges[func_node.get_name()] = {}
        
        if node.name == "effective_principals" and self.qika_effective_principals:
            self.qika_effective_principals = 2
        
        # Same remarks as for ClassDef above.
        #
        self.associate_node(func_node, node, self.filename)
        self.set_value(node.name, new_value=func_node)

        # Enter the function scope
        #
        self.name_stack.append(node.name)
        inner_ns = self.get_node_of_current_namespace().get_name()
        self.scope_stack.append(self.scopes[inner_ns])
        self.context_stack.append("FunctionDef %s" % (node.name))


        # Capture which names correspond to function args.
        #
        self.generate_args_nodes(node.args, inner_ns)

        # self_name is just an ordinary name in the method namespace, except
        # that its value is implicitly set by Python when the method is called.
        #
        # Bind self_name in the function namespace to its initial value,
        # i.e. the current class. (Class, because Pyan cares only about
        # object types, not instances.)
        #
        # After this point, self_name behaves like any other name.
        #
        if self_name is not None:
            class_node = self.get_current_class()
            self.scopes[inner_ns].defs[self_name] = class_node
            self.logger.info('Method def: setting self name "%s" to %s' % (self_name, class_node))

        # record bindings of args to the given default values, if present
        self.analyze_arguments(node.args)
        
        # Analyze the function body
        
        self.functional_info[func_node.get_name()] = {"annotation": ''}
        # 如果body中首个stmt是注释，则将其作为功能注释
        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr):
            
            if isinstance(node.body[0].value, ast.Constant) and type(node.body[0].value.s)==str:
                self.functional_info[func_node.get_name()]["annotation"] = node.body[0].value.s
                self.functional_info[func_node.get_name()]["annotation_begin"] = node.body[0].lineno 
                self.functional_info[func_node.get_name()]["annotation_end"] = node.body[0].end_lineno
            elif isinstance(node.body[0].value, ast.Str):
                self.functional_info[func_node]["annotation"] = node.body[0].value
                self.functional_info[func_node.get_name()]["annotation_begin"] = node.body[0].lineno 
                self.functional_info[func_node.get_name()]["annotation_end"] = node.body[0].end_lineno
             
        
        self.functional_info[func_node.get_name()]["body_begin"] = -1
        self.functional_info[func_node.get_name()]["body_end"] = -1
        begin_num = 1 if self.functional_info[func_node.get_name()]["annotation"] != '' else 0
        if len(node.body) > begin_num:
            self.functional_info[func_node.get_name()]["body_begin"] = node.body[begin_num].lineno
            self.functional_info[func_node.get_name()]["body_end"] = node.body[-1].end_lineno
    
    
        if func_node.get_name() == 'xmnlp.module.Module.save':
            qika = 1

        for stmt in node.body:
            self.visit(stmt)
            
        # qika 将函数返回类型加入scope。
        if node.returns is not None:
            return_node = node.returns.value if isinstance(node.returns, ast.Constant) else node.returns
            if return_node is not None:
                if isinstance(node.returns, ast.Constant):
                    candidate_node, sc = self.get_value(return_node)
                    candidate_name = return_node
                    candidate_type = candidate_node.get_type() if isinstance(candidate_node, Node) else None
                else:
                    self.visit(return_node)
                    candidate_node = self.last_value
                    candidate_type = self.last_type
                    candidate_name = candidate_node.name if isinstance(candidate_node, Node) else None
                    
                
                if candidate_type is not None:
                    return_type = candidate_type
                    self.scopes[inner_ns].set_Return(return_type)
                    self.logger.debug("Set Return of %s as %s" % (inner_ns, return_type))
                elif candidate_name is not None:
                    return_type = candidate_name
                    if self.scopes[inner_ns].Return is None:
                        self.scopes[inner_ns].set_Return(return_type)
                        self.logger.debug("Set Return of %s as %s" % (inner_ns, return_type))

        # Exit the function scope
        #
        self.context_stack.pop()
        self.scope_stack.pop()
        self.name_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)  # TODO: alias for now; tag async functions in output in a future version?

    def visit_Lambda(self, node):
        # TODO: avoid lumping together all lambdas in the same namespace.
        self.logger.debug("Lambda, %s:%s" % (self.filename, node.lineno))
        with ExecuteInInnerScope(self, "lambda"):
            inner_ns = self.get_node_of_current_namespace().get_name()
            self.generate_args_nodes(node.args, inner_ns)
            self.analyze_arguments(node.args)
            self.visit(node.body)  # single expr
    
    def visit_Return(self, node):
        """qika 通過對返回值的分析，猜測函數的返回類型"""
        """为保证准确性，关闭猜测"""
        inner_ns = self.get_node_of_current_namespace().get_name()
        self.last_value = None
        self.generic_visit(node)
        type_ns, type_name = self.split(self.last_type)
        return_node = self.get_node(type_ns, type_name)
        if self.last_type and return_node and return_node.defined:
            self.scopes[inner_ns].set_Return(return_node.get_type())
            pass
            
            

    def generate_args_nodes(self, ast_args, inner_ns):
        """Capture which names correspond to function args.

        In the function scope, set them to a nonsense Node,
        to prevent leakage of identifiers of matching name
        from the enclosing scope (due to the local value being None
        until we set it to this nonsense Node).

        ast_args: node.args from a FunctionDef or Lambda
        inner_ns: namespace of the function or lambda, for scope lookup
        """
        sc = self.scopes[inner_ns]
        # As the name of the nonsense node, we can use any string that
        # is not a valid Python identifier.
        #
        # It has no sensible flavor, so we leave its flavor unspecified.
        nonsense_node = self.get_node(inner_ns, "^^^argument^^^", None)
        # args, vararg (*args), kwonlyargs, kwarg (**kwargs)
        for a in ast_args.args:  # positional
            sc.defs[a.arg] = nonsense_node
        if ast_args.vararg is not None:  # *args if present
            sc.defs[ast_args.vararg] = nonsense_node
        for a in ast_args.kwonlyargs:  # any after *args or *
            sc.defs[a.arg] = nonsense_node
        if ast_args.kwarg is not None:  # **kwargs if present
            sc.defs[ast_args.kwarg] = nonsense_node

    def analyze_arguments(self, ast_args):
        """Analyze an arguments node of the AST.

        Record bindings of args to the given default values, if present.

        Used for analyzing FunctionDefs and Lambdas."""
        # https://greentreesnakes.readthedocs.io/en/latest/nodes.html?highlight=functiondef#arguments
        
        # qika：如果参数有类型注释，则标记为对应类型
        for tgt in ast_args.args:
            if tgt.annotation is not None:
                self.visit(tgt)
        
        if ast_args.defaults:
            n = len(ast_args.defaults)
            for tgt, val in zip(ast_args.args[-n:], ast_args.defaults):
                targets = sanitize_exprs(tgt)
                values = sanitize_exprs(val)
                self.analyze_binding(targets, values)
        
                    
        if ast_args.kw_defaults:
            n = len(ast_args.kw_defaults)
            for tgt, val in zip(ast_args.kwonlyargs, ast_args.kw_defaults):
                if val is not None:
                    targets = sanitize_exprs(tgt)
                    values = sanitize_exprs(val)
                    self.analyze_binding(targets, values)
                    
    def visit_Import(self, node):
        self.logger.debug("Import %s, %s:%s" % ([format_alias(x) for x in node.names], self.filename, node.lineno))

        # TODO: add support for relative imports (path may be like "....something.something")
        # https://www.python.org/dev/peps/pep-0328/#id10

        for import_item in node.names:  # the names are modules
            self.analyze_module_import(import_item, node)
        
        pass

    def visit_ImportFrom(self, node):
        self.logger.debug(
            "ImportFrom: from %s import %s, %s:%s"
            % (node.module, [format_alias(x) for x in node.names], self.filename, node.lineno)
        )
        # Pyan needs to know the package structure, and how the program
        # being analyzed is actually going to be invoked (!), to be able to
        # resolve relative imports correctly.
        #
        # As a solution, we register imports here and later, when all files have been parsed, resolve them.
        from_node = self.get_node_of_current_namespace()
        if node.module is None:  # resolve relative imports 'None' such as "from . import foo"
            self.logger.debug(
                "ImportFrom (original) from %s import %s, %s:%s"
                % ("." * node.level, [format_alias(x) for x in node.names], self.filename, node.lineno)
            )
            tgt_level = node.level
            current_module_namespace = self.module_name.rsplit(".", tgt_level)[0]
            tgt_name = current_module_namespace
            self.logger.debug(
                "ImportFrom (resolved): from %s import %s, %s:%s"
                % (tgt_name, [format_alias(x) for x in node.names], self.filename, node.lineno)
            )
        elif node.level != 0:  # resolve from ..module import foo
            self.logger.debug(
                "ImportFrom (original): from %s import %s, %s:%s"
                % (node.module, [format_alias(x) for x in node.names], self.filename, node.lineno)
            )
            tgt_level = node.level
            current_module_namespace = self.module_name.rsplit(".", tgt_level)[0]
            tgt_name = current_module_namespace + "." + node.module
            self.logger.debug(
                "ImportFrom (resolved): from %s import %s, %s:%s"
                % (tgt_name, [format_alias(x) for x in node.names], self.filename, node.lineno)
            )
        else:
            if node.module == "asciimatics.constants":
                qika = 1
            cut_module_namespace = self.module_name.rsplit(".", 1)[0]+"." if "." in self.module_name else "" 
            module_name = node.module.split(".")[0]  # normal from module.submodule import foo
            if module_name in cut_module_namespace:
                module_index = cut_module_namespace.rfind(module_name)
                if module_index>0:
                    tgt_name = cut_module_namespace[:module_index]+node.module
                elif module_index==0:
                    tgt_name = node.module
            else:
                tgt_name = node.module


        # link each import separately
        for alias in node.names:
            
            # if there is alias, add extra edge between alias and node
            if alias.asname is not None:
                alias_name = alias.asname
            else:
                alias_name = alias.name
            
            # 首先定义本空间下的虚拟节点 import_node
            from_node = self.get_node_of_current_namespace()
            current_namespace = from_node.get_name()
            import_node = self.get_node(current_namespace, alias_name, flavor=Flavor.IMPORTEDITEM)

            # 假设import一个虚拟类，定义对应的虚拟module节点
            mod_node = self.get_node(tgt_name, alias.name, flavor=Flavor.IMPORTEDITEM)
            mod_node.set_type(mod_node.get_name())
            
            # 将import_node的类型设置为虚拟类路径，并且将其的值设置为虚拟类节点
            import_node.set_type(mod_node.get_name())
            import_node.set_value(mod_node)
            
            
            # 尝试获取import的真实类对应的节点
            if tgt_name+"."+alias.name in self.module_to_filename:
                mod_node = self.get_node(tgt_name, alias.name, flavor=Flavor.IMPORTEDITEM)
                if mod_node.defined:
                    # 将import_node的类型设置为真实类节点路径，并且将其的值设置为真实类节点
                    self.logger.debug("Set type of %s to %s" % (import_node,mod_node))
                    import_node.set_value(mod_node)
                    import_node.set_type(mod_node.get_type())

            elif tgt_name in self.module_to_filename:
                mod_node = self.find_scope_def_node(tgt_name, alias.name)
                if mod_node and mod_node.defined:
                    # 将import_node的类型设置为真实类节点路径，并且将其的值设置为真实类节点
                    self.logger.debug("Set type of %s to %s" % (import_node,mod_node))
                    import_node.set_value(mod_node)
                    import_node.set_type(mod_node.get_type())
                else:
                    candidate_path = tgt_name+"."+"__init__"
                    mod_node = self.find_scope_def_node(candidate_path, alias.name)
                    if mod_node and mod_node.defined:
                        # 将import_node的类型设置为真实类节点路径，并且将其的值设置为真实类节点
                        self.logger.debug("Set type of %s to %s" % (import_node,mod_node))
                        import_node.set_value(mod_node)
                        import_node.set_type(mod_node.get_type())

            if from_node.get_name() not in self.import_uses_edges:
                self.import_uses_edges[from_node.get_name()] = {}
            self.import_uses_edges[from_node.get_name()][alias_name] = mod_node if isinstance(mod_node, Node) and mod_node.defined else self.get_node(tgt_name, alias.name, flavor=Flavor.IMPORTEDITEM)
                    
            self.logger.debug("Use as ImportFrom from %s to %s" % (from_node, import_node))
            if self.add_uses_edge(from_node, import_node):
                
                self.logger.info("New edge added for Use from %s to %s" % (from_node, import_node))
            
            
            self.logger.info("From setting name %s to %s" % (alias_name, import_node.get_value()))
            self.set_value(alias_name, new_value=import_node.get_value(), defined = False)  # set node to be discoverable in module
            
            
        pass
            
                

    def analyze_module_import(self, import_item, ast_node):
        """Analyze a names AST node inside an Import or ImportFrom AST node.

        This handles the case where the objects being imported are modules.

        import_item: an item of ast_node.names
        ast_node: for recording source location information
        """
        src_name = import_item.name  # what is being imported

        # mark the use site
        #
        # where it is being imported to, i.e. the **user**
        from_node = self.get_node_of_current_namespace()
        current_namespace = from_node.get_name()
        # the thing **being used** (under the asname, if any)
        mod_node = self.get_node("", src_name, flavor=Flavor.IMPORTEDITEM)
        mod_node.set_type(mod_node.get_name())
        mod_node.set_value(mod_node)
        # if there is alias, add extra edge between alias and node
        if import_item.asname is not None:
            alias_name = import_item.asname
        else:
            alias_name = mod_node.name

        if from_node.get_name() not in self.import_uses_edges:
            self.import_uses_edges[from_node.get_name()] = {}
        self.import_uses_edges[from_node.get_name()][alias_name] = mod_node
        if self.add_uses_edge(from_node, mod_node):
            self.logger.info("New edge added for Use import %s in %s" % (mod_node, from_node))
        
        import_node = self.get_node(current_namespace, alias_name, flavor=Flavor.IMPORTEDITEM)
        import_node.set_value(mod_node)
        import_node.set_type(mod_node.get_name())
        
        self.set_value(alias_name, new_value=import_node, defined=False)  # set node to be discoverable in module
        self.logger.info("From setting name %s to %s" % (alias_name, mod_node))
        
        pass

    # Edmund Horner's original post has info on what this fixed in Python 2.
    # https://ejrh.wordpress.com/2012/01/31/call-graphs-in-python-part-2/
    #
    # Essentially, this should make '.'.join(...) see str.join.
    # Pyan3 currently handles that in resolve_attribute() and get_attribute().
    #
    # Python 3.4 does not have ast.Constant, but 3.6 does.
    # TODO: actually test this with Python 3.6 or later.
    #
    def visit_Constant(self, node):
        self.logger.debug("Constant %s, %s:%s" % (node.value, self.filename, node.lineno))
        t = type(node.value)
        ns = self.get_node_of_current_namespace().get_name()
        tn = t.__name__
        constant_node = self.get_node("", tn, None, flavor=Flavor.NAME)
        constant_node.set_type(tn)
        self.last_value = constant_node
        self.last_type = tn

    # attribute access (node.ctx determines whether set (ast.Store) or get (ast.Load))
    def visit_Attribute(self, node):
        objname = get_ast_node_name(node.value)
        if objname == "bz2.BZ2File":
            qika = 1
        self.logger.debug(
            "Attribute %s of %s in context %s, %s:%s" % (node.attr, objname, type(node.ctx), self.filename, node.lineno)
        )

        if isinstance(node.ctx, ast.Store):
            new_value = self.last_value
            new_type = self.last_type
            try:
                if self.set_attribute(node, new_value, new_type):
                    self.logger.info("setattr %s on %s to %s" % (node.attr, objname, new_value))
            except UnresolvedSuperCallError:
                # Trying to set something belonging to an unresolved super()
                # of something; just ignore this attempt to setattr.
                return

        elif isinstance(node.ctx, ast.Load):
            try:
                obj_node, attr_node = self.get_attribute(node)
                self.last_value = attr_node
                self.last_type = attr_node.get_type()
            except UnresolvedSuperCallError:
                self.last_value = None
                self.last_type = None
                return

            # Both object and attr known.
            if isinstance(attr_node, Node):
                from_node = self.get_node_of_current_namespace()
                # remove resolved wildcard from current site to <Node *.attr>
                if attr_node.namespace is not None:
                    self.remove_wild(from_node, attr_node, node.attr)

        pass
    
    def lookup(self, ns, attr_name):
            if ns in self.scopes:
                sc = self.scopes[ns]
                if attr_name in sc.defs:
                    return sc.defs[attr_name]
            return None
    
    # name access (node.ctx determines whether set (ast.Store) or get (ast.Load))
    def visit_arg(self, node):
        self.logger.debug("arg %s, %s:%s" % (node.arg, self.filename, node.lineno))

        # 获取参数类型标注
        arg_type = None
        if node.annotation:
            tgt_type = node.annotation
            if isinstance(tgt_type, ast.Constant):
                tgt_name = tgt_type.value
                type_node, type_node_sc = self.get_value(tgt_name)  # resolves "self" if needed
                arg_type = type_node.get_type() if type_node else None
            else:
                self.visit(tgt_type)
                arg_type = self.last_type

        # 定义当前命名空间下的参数节点，将其作为scope中此名称的值
        from_node = self.get_node_of_current_namespace()
        current_namespace = from_node.get_name()
        arg_node = self.get_node(current_namespace, node.arg, node, flavor=Flavor.NAME)
        arg_node.defined = True
        if arg_type is not None:
            arg_node.set_type(arg_type)
        
        self.set_value(node.arg, new_value=arg_node)
        self.last_value = arg_node
        self.last_type = arg_node.get_type()

    # name access (node.ctx determines whether set (ast.Store) or get (ast.Load))
    def visit_Name(self, node):
        self.logger.debug("Name %s in context %s, %s:%s" % (node.id, type(node.ctx), self.filename, node.lineno))

        if isinstance(node.ctx, ast.Store):
            # when we get here, self.last_value has been set by visit_Assign()
            self.set_value(node.id, self.last_value, self.last_type, node = node)
            

        # A name in a load context is a use of the object the name points to.
        elif isinstance(node.ctx, ast.Load):
            tgt_name = node.id
            name_node, name_ns = self.get_value(tgt_name) 
            
            if tgt_name == "Everyone" and self.qika_effective_principals == 2:
                self.qika_effective_principals = 3
            
            from_node = self.get_node_of_current_namespace()
            
            # 特殊处理类型名（如int, str, float等）
            type_names = ['int', 'str', 'float', 'bool', 'object', 'list', 'dict', 'tuple']
            if name_node is None and tgt_name in type_names:
                name_node = self.get_node("", tgt_name, flavor=Flavor.NAME)
                name_node.set_type(tgt_name)
                name_node.defined = False

            # qika 特殊处理__name__，将其作为字符串常量
            if name_node is None and tgt_name == '__name__':
                name_node = self.get_node("", tgt_name, flavor=Flavor.NAME)
                name_node.set_type('str')
                name_node.defined = False

            current_class = self.get_current_class()
            


            # 如name不是当前类self或者namespace中定义的对象，添加对name的use边
            if from_node and (current_class is None or name_node is not current_class) and name_ns != from_node.get_name() and name_node: 
                # 如果name是一个被import的名称，则保留import信息
                for prefix in self.scope_stack[::-1]:
                    prefix = prefix.path
                    if prefix in self.import_uses_edges and tgt_name in self.import_uses_edges[prefix]:
                        import_node = self.import_uses_edges[prefix][tgt_name]
                        if isinstance(import_node, Node) and import_node.get_type() == name_node.get_type():
                            if from_node.get_name() not in self.import_uses_edges:
                                self.import_uses_edges[from_node.get_name()] = {}
                            self.import_uses_edges[from_node.get_name()][tgt_name] = import_node
                            break

                # 如果name在name_ns下是一个被defined的节点，则认为其是引用了一个变量而非类型，添加true_type=False的use边
                if name_node.defined:
                    self.logger.debug("Use name from %s to %s (use as a variable)" % (from_node, name_node))
                    if self.add_uses_edge(from_node, name_node, true_type=False):
                        self.logger.info(
                            "New edge added for Use from %s to %s (use a variable)" % (from_node, name_node)
                        ) 
                # 不然，认为name不是在name_ns下定义的，而是引入的一个外部类名称或函数名称，即使用了一个真实类或函数类型，添加true_type=True的use边
                else:
                    self.logger.debug("Use name from %s to %s (use as a type)" % (from_node, name_node))
                    if self.add_uses_edge(from_node, name_node):
                        self.logger.info(
                            "New edge added for Use from %s to %s (use a type name)" % (from_node, name_node.get_type())
                        ) 

            # 如果name是list或dict，则考虑“列表的类型是其中元素的类型”，不赋值
            if name_node is not None and name_node.get_type() in ['list', 'dict', 'tuple']:
                pass
            else:
                self.last_value = name_node
                self.last_type = name_node.get_type() if isinstance(name_node, Node) else None

    def visit_Assign(self, node):
        # - chaining assignments like "a = b = c" produces multiple targets
        # - tuple unpacking works as a separate mechanism on top of that (see analyze_binding())
        #
        if len(node.targets) > 1:
            self.logger.debug("Assign (chained with %d outputs)" % (len(node.targets)))

        # TODO: support lists, dicts, sets (so that we can recognize calls to their methods)
        # TODO: begin with supporting empty lists, dicts, sets
        # TODO: need to be more careful in sanitizing; currently destroys a bare list

        values = sanitize_exprs(node.value)  # values is the same for each set of targets
        
        # qika: 如果是将一个列表整体赋值给一个变量，则将列表中的元素类型作为变量类型
        for targets in node.targets:
            targets = sanitize_exprs(targets)
            self.logger.debug(
                "Assign %s %s, %s:%s"
                % (
                    [get_ast_node_name(x) for x in targets],
                    [get_ast_node_name(x) for x in values],
                    self.filename,
                    node.lineno,
                )
            )
            self.analyze_binding(targets, values)
        
        pass

    def visit_AnnAssign(self, node):  # PEP 526, Python 3.6+
        target = sanitize_exprs(node.target)
        self.last_value = None
        self.last_type = None
        if node.value is not None:
            value = sanitize_exprs(node.value)
            if len(value) == 0:
                value = [node.value]
            self.logger.debug(
                "AnnAssign %s %s, %s:%s"
                % (get_ast_node_name(target[0]), get_ast_node_name(value), self.filename, node.lineno)
            )
            self.analyze_binding(target, value)
            
        else:  # just a type declaration
            self.logger.debug(
                "AnnAssign %s <no value>, %s:%s" % (get_ast_node_name(target[0]), self.filename, node.lineno)
            )
            self.last_value = None
            self.visit(target[0])

        # qika：如果有类型注释，则标记为对应类型
        if node.annotation is not None:
            tgt_type = node.annotation
            if isinstance(tgt_type, ast.Constant):
                type_node, type_node_sc = self.get_value(tgt_type.value)  # resolves "self" if needed
                var_type = type_node.get_type() if type_node else None
            else:
                self.visit(tgt_type)
                var_type = self.last_type
            
            self.logger.debug(
                "AnnAssign %s type %s"
                % (get_ast_node_name(target[0]), var_type)
            )
                
            for tgt in target:
                self.last_type = var_type
                self.last_value = None
                self.visit(tgt)

    def visit_AugAssign(self, node):
        targets = sanitize_exprs(node.target)
        values = sanitize_exprs(node.value)  # values is the same for each set of targets

        self.logger.debug(
            "AugAssign %s %s %s, %s:%s"
            % (
                [get_ast_node_name(x) for x in targets],
                type(node.op),
                [get_ast_node_name(x) for x in values],
                self.filename,
                node.lineno,
            )
        )

        # TODO: maybe no need to handle tuple unpacking in AugAssign? (but simpler to use the same implementation)
        self.analyze_binding(targets, values)

    # for() is also a binding form.
    #
    # (Without analyzing the bindings, we would get an unknown node for any
    #  use of the loop counter(s) in the loop body. This would have confusing
    #  consequences in the expand_unknowns() step, if the same name is
    #  in use elsewhere.)
    #
    def visit_For(self, node):
        self.logger.debug("For-loop, %s:%s" % (self.filename, node.lineno))

        targets = sanitize_exprs(node.target)
        values = sanitize_exprs(node.iter)
        self.analyze_binding(targets, values)

        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
    
    def visit_Subscript(self, node):
        self.logger.debug("Subscript, %s:%s" % (self.filename, node.lineno))
        
        self.visit(node.slice)
        # 如果列表名为Optional，则将slice的类型作为列表的类型
        if isinstance(node.value, ast.Name) and node.value.id == 'Optional':
            pass
        else:
            # 默认列表的类型为列表中元素的类型，也将其作为当前subscript的类型
            self.visit(node.value)
    
    def visit_List(self, node):
        # 此为“[1,2,3]”等形式的无名列表，只需visit其中元素
        self.logger.debug("List, %s:%s" % (self.filename, node.lineno))
        for elt in node.elts:
            self.visit(elt)

    def visit_AsyncFor(self, node):
        self.visit_For(node)  # TODO: alias for now; tag async for in output in a future version?

    def visit_ListComp(self, node):
        self.logger.debug("ListComp, %s:%s" % (self.filename, node.lineno))
        self.analyze_comprehension(node, "listcomp")

    def visit_SetComp(self, node):
        self.logger.debug("SetComp, %s:%s" % (self.filename, node.lineno))
        self.analyze_comprehension(node, "setcomp")

    def visit_DictComp(self, node):
        self.logger.debug("DictComp, %s:%s" % (self.filename, node.lineno))
        self.analyze_comprehension(node, "dictcomp", field1="key", field2="value")

    def visit_GeneratorExp(self, node):
        self.logger.debug("GeneratorExp, %s:%s" % (self.filename, node.lineno))
        self.analyze_comprehension(node, "genexpr")

    def analyze_comprehension(self, node, label, field1="elt", field2=None):
        # The outermost iterator is evaluated in the current scope;
        # everything else in the new inner scope.
        #
        # See function symtable_handle_comprehension() in
        #   https://github.com/python/cpython/blob/master/Python/symtable.c
        # For how it works, see
        #   https://stackoverflow.com/questions/48753060/what-are-these-extra-symbols-in-a-comprehensions-symtable
        # For related discussion, see
        #   https://bugs.python.org/issue10544
        
        if self.filename in ["/home/lijia/Context_benchmark/Source_Code_Release/Internet/sumy/sumy/summarizers/reduction.py"]:
            qika = 1
        
        gens = node.generators  # tuple of ast.comprehension
        outermost = gens[0]
        moregens = gens[1:] if len(gens) > 1 else []

        outermost_iters = sanitize_exprs(outermost.iter)
        outermost_targets = sanitize_exprs(outermost.target)
        for expr in outermost_iters:
            self.visit(expr)  # set self.last_value (to something and hope for the best)

        with ExecuteInInnerScope(self, label):
            for expr in outermost.ifs:
                self.visit(expr)
                
            for expr in outermost_targets:
                    self.visit(expr)  # use self.last_value

            # TODO: there's also an is_async field we might want to use in a future version of Pyan.
            for gen in moregens:
                targets = sanitize_exprs(gen.target)
                values = sanitize_exprs(gen.iter)
                self.analyze_binding(targets, values)
                for expr in gen.ifs:
                    self.visit(expr)

            self.visit(getattr(node, field1))  # e.g. node.elt
            if field2:
                self.visit(getattr(node, field2))

    def visit_Call(self, node):
        
        def lookup(ns, attr_name):
            if ns in self.scopes:
                sc = self.scopes[ns]
                if attr_name in sc.defs:
                    return sc.defs[attr_name]
            return None
        
        self.logger.debug("Call %s, %s:%s" % (get_ast_node_name(node.func), self.filename, node.lineno))

        # visit args to detect uses
        for arg in node.args:
            self.visit(arg)
        for kw in node.keywords:
            self.visit(kw.value)
            

        # see if we can predict the result
        try:
            result_node = self.resolve_builtins(node)
        except UnresolvedSuperCallError:
            result_node = None

        if isinstance(result_node, Node):  # resolved result
            self.last_value = result_node
            self.last_type = result_node.get_type()

            from_node = self.get_node_of_current_namespace()
            to_node = result_node
            self.logger.debug("Use from %s to %s (via resolved call to built-ins)" % (from_node, to_node))
            if self.add_uses_edge(from_node, to_node):
                self.logger.info(
                    "New edge added for Use from %s to %s (via resolved call to built-ins)" % (from_node, to_node)
                )

        else:  # generic function call
            # Visit the function name part last, so that inside a binding form,
            # it will be left standing as self.last_value.
            self.last_value = None
            self.last_type = None
            self.visit(node.func)
            
            # If self.last_value matches a known class i.e. the call was of the
            # form MyClass(), add a uses edge to MyClass.__init__().
            #
            # We need to do this manually, because there is no text "__init__"
            # at the call site.
            #
            # In this lookup to self.class_base_ast_nodes we don't care about
            # the AST nodes; the keys just conveniently happen to be the Nodes
            # of known classes.
            #
            
            # 首先定义当前命名空间下的函数调用虚拟节点 call_node
            from_node = self.get_node_of_current_namespace()
            current_namespace = from_node.get_name()
            # call_name = get_ast_node_name(node.func)
            if self.last_value is not None:
                call_node = self.get_node(self.last_value.get_name(), "()", None, flavor=Flavor.NAME)
                call_node.set_value(self.last_value)
            else:
                call_node = self.get_node(current_namespace, "NUK_CALL", None, flavor=Flavor.NAME)
            if self.last_type is not None:
                call_node.set_type(self.last_type)
                
            if self.last_value in self.class_base_ast_nodes:
                class_node = self.last_value
                
                if class_node:
                    # first try directly in object's ns (this works already in pass 1)
                    ns = class_node.get_name()
                    value_node = lookup(ns, "__init__")
                    # next try ns of each ancestor (this works only in pass 2,
                    # after self.mro has been populated)
                    #
                    
                    if value_node is None and class_node in self.mro:
                        base_node = None
                        for up in self.mro[class_node]:  # the first element is always obj itself
                            base_node = up
                            ns = base_node.get_name()
                            value_node = lookup(ns, "__init__")
                            if value_node is not None:
                                break
                        class_node = base_node
                # qika: 这里会导致第三方引入类的构造函数被认为是虚拟节点（未在项目中被定义），连virtual_use边
                to_node = self.get_node(class_node.get_name(), "__init__", None, flavor=Flavor.METHOD)
                to_node.set_type(to_node.get_name())
                self.logger.debug("Use from %s to %s (call creates an instance)" % (from_node, to_node))
                if self.add_uses_edge(from_node, to_node):
                    self.logger.info(
                        "New edge added for Use from %s to %s (call creates an instance)" % (from_node, to_node)
                    )
            
            # qika: 增加对Call的返回类型的猜测，使last_value为返回类型
            # qika TODO: 无法用于猜测super(), 因为super()不需声明,因此self.last_value为None
            if call_node.get_type() is not None:
                inner_ns = call_node.get_type()
                if inner_ns in self.scopes:
                    sc = self.scopes[inner_ns]
                    if sc.Return is not None:
                        self.last_type = sc.Return
                        call_node.set_type(sc.Return)
            
            # qika 對於cast(type, value)的處理,分析type获得返回类型
            if isinstance(call_node, Node) and call_node.name == 'cast':
                self.visit(node.args[0])
                call_node.set_type(self.last_type)
                
            # qika 对于get_attr(obj, attr)的处理，分析obj.attr获得返回类型
            if isinstance(node.func, ast.Name) and node.func.id == 'getattr':
                obj_node = node.args[0]
                attr_node = node.args[1]
                if isinstance(attr_node, ast.Constant):
                    attr_name = attr_node.value
                elif isinstance(attr_node, ast.Name):
                    attr_name = attr_node.id
                else:
                    self.visit(attr_node)
                    attr_name = self.last_value.get_name() if isinstance(self.last_value, Node) else None
                
                self.visit(obj_node)
                obj_node = self.last_value
                if isinstance(obj_node, Node) and obj_node.get_type() is not None:
                    obj_type = obj_node.get_type()
                    if obj_type in self.scopes:
                        sc = self.scopes[obj_type]
                        if attr_name in sc.defs and sc.defs[attr_name].get_type() is not None:
                            attr_node = sc.defs[attr_name]
                            call_node.set_type(attr_node.get_type())
                            if attr_node.defined:
                                if self.add_uses_edge(from_node, attr_node, true_type=False):
                                    self.logger.info(
                                        "New edge added for Use from %s to %s (use as a attribute)" % (from_node, attr_node)
                                    ) 
                            else:
                                if self.add_uses_edge(from_node, attr_node):
                                    self.logger.info(
                                        "New edge added for Use from %s to %s (use as a type)" % (from_node, attr_node)
                                    ) 
                                    
                                
                
            
            # 如果针对函数f()无法获得返回类型
            if call_node.get_type() is None:
                # 如果函数f()本身对应真实节点Nf，则设置type为(Nf.get_name())()
                if call_node.get_value() is not None:
                    call_node.set_type(call_node.get_value().get_name())
            
            self.last_type = call_node.get_type()
            self.last_value = call_node
                
                

    def visit_With(self, node):
        self.logger.debug("With (context manager), %s:%s" % (self.filename, node.lineno))

        def add_uses_enter_exit_of(graph_node):
            # add uses edges to __enter__ and __exit__ methods of given Node
            if isinstance(graph_node, Node):
                from_node = self.get_node_of_current_namespace()
                withed_obj_node = graph_node

                self.logger.debug("Use from %s to With %s" % (from_node, withed_obj_node))
                """ for methodname in ("__enter__", "__exit__"):
                    to_node = self.get_node(withed_obj_node.get_name(), methodname, None, flavor=Flavor.METHOD)
                    to_node.set_type(to_node.get_name())
                    if self.add_uses_edge(from_node, to_node):
                        self.logger.info("New edge added for Use from %s to %s" % (from_node, to_node))"""

        for withitem in node.items:
            expr = withitem.context_expr
            vars = withitem.optional_vars

            # XXX: we currently visit expr twice (again in analyze_binding()) if vars is not None
            self.last_value = None
            self.last_type = None
            self.visit(expr)
            if isinstance(self.last_value, Node):
                obj_ns, obj_name = self.split(self.last_value.get_type())
                obj_node = self.get_node(obj_ns, obj_name)
                add_uses_enter_exit_of(obj_node)

            if vars is not None:
                # bind optional_vars
                #
                # TODO: For now, we support only the following (most common) case:
                #  - only one binding target, vars is ast.Name
                #    (not ast.Tuple or something else)
                #  - the variable will point to the object that was with'd
                #    (i.e. we assume the object's __enter__() method
                #     to finish with "return self")
                #
                if isinstance(vars, ast.Name):
                    self.analyze_binding(sanitize_exprs(vars), sanitize_exprs(expr))
                else:
                    self.visit(vars)  # just capture any uses on the With line itself
            self.last_value = None
            self.last_type = None

        for stmt in node.body:
            self.visit(stmt)

    ###########################################################################
    # Analysis helpers

    def analyze_functiondef(self, ast_node):
        """Analyze a function definition.

        Visit decorators, and if this is a method definition, capture the name
        of the first positional argument to denote "self", like Python does.

        Return (self_name, flavor), where self_name the name representing self,
        or None if not applicable; and flavor is a Flavor, specifically one of
        FUNCTION, METHOD, STATICMETHOD or CLASSMETHOD."""

        if not isinstance(ast_node, (ast.AsyncFunctionDef, ast.FunctionDef)):
            raise TypeError("Expected ast.FunctionDef; got %s" % (type(ast_node)))

        # Visit decorators
        self.last_value = None
        deco_names = []
        for deco in ast_node.decorator_list:
            self.visit(deco)  # capture function name of decorator (self.last_value hack)
            deco_node = self.last_value
            if isinstance(deco_node, Node):
                deco_names.append(deco_node.name)
            self.last_value = None

        # Analyze flavor
        in_class_ns = self.context_stack[-1].startswith("ClassDef")
        if not in_class_ns:
            flavor = Flavor.FUNCTION
        else:
            if "property" in deco_names:
                flavor = Flavor.PROPERTYMETHOD
            elif "staticmethod" in deco_names:
                flavor = Flavor.STATICMETHOD
            elif "classmethod" in deco_names:
                flavor = Flavor.CLASSMETHOD
            else:  # instance method
                flavor = Flavor.METHOD

        # Get the name representing "self", if applicable.
        #
        # - ignore static methods
        # - ignore functions defined inside methods (this new FunctionDef
        #   must be directly in a class namespace)
        #
        if flavor in (Flavor.PROPERTYMETHOD, Flavor.METHOD, Flavor.CLASSMETHOD):
            # We can treat instance methods and class methods the same,
            # since Pyan is only interested in object types, not instances.
            all_args = ast_node.args  # args, vararg (*args), kwonlyargs, kwarg (**kwargs)
            posargs = all_args.args
            if len(posargs):
                self_name = posargs[0].arg
                return self_name, flavor

        return None, flavor

    def analyze_binding(self, targets, values):
        """Generic handler for binding forms. Inputs must be sanitize_exprs()d."""
        
        # qika：v2中修改node赋值方式，将target.type标记为value。如果value是真实节点，则将value赋值给node。

        # Before we begin analyzing the assignment, clean up any leftover self.last_value.
        self.last_value = None
        self.last_type = None

        if len(targets) == len(values):  # handle correctly the most common trivial case "a1,a2,... = b1,b2,..."
            captured_values = []
            captured_types = []
            for value in values:
                self.visit(value)  # RHS -> set self.last_value
                captured_values.append(self.last_value)
                captured_types.append(self.last_type)
                self.last_value = None
                self.last_type = None
            for tgt, val, typ in zip(targets, captured_values, captured_types):
                self.last_value = val
                self.last_type = typ
                self.visit(tgt)  # LHS, name in a store context
            self.last_value = None
            self.last_type = None
        else:  # FIXME: for now, do the wrong thing in the non-trivial case
            # old code, no tuple unpacking support
            captured_values = []
            captured_types = []
            for value in values:
                self.visit(value)  # set self.last_value to **something** on the RHS and hope for the best
                captured_values.append(self.last_value)
                captured_types.append(self.last_type)
            for tgt in targets:  # LHS, name in a store context
                if isinstance(tgt, ast.Name):
                    self.last_value = captured_values[0] if len(captured_values) else None
                    self.last_type = captured_types[0] if len(captured_types) else None
                    self.visit(tgt)
                else:
                    self.last_value = None 
                    self.last_type = None
                    self.visit(tgt)
            self.last_value = None
            self.last_type = None
        
        pass

    def resolve_builtins(self, ast_node):
        """Resolve those calls to built-in functions whose return values
        can be determined in a simple manner.

        Currently, this supports:

          - str(obj), repr(obj) --> obj.__str__, obj.__repr__

          - super() (any arguments ignored), which works only in pass 2,
            because the MRO is determined between passes.

        May raise UnresolvedSuperCallError, if the call is to super(),
        but the result cannot be (currently) determined (usually because either
        pass 1, or some relevant source file is not in the analyzed set).

        Returns the Node the call resolves to, or None if not determined.
        
        qika:这里包含对内置函数返回值的类型判断
        """
        if not isinstance(ast_node, ast.Call):
            raise TypeError("Expected ast.Call; got %s" % (type(ast_node)))

        func_ast_node = ast_node.func  # expr
        if isinstance(func_ast_node, ast.Name):
            funcname = func_ast_node.id
            if funcname == "super":
                class_node = self.get_current_class()
                self.logger.debug("Resolving super() of %s" % (class_node))
                if class_node in self.mro:
                    # Our super() class is the next one in the MRO.
                    #
                    # Note that we consider only the **static type** of the
                    # class itself. The later elements of the MRO - important
                    # for resolving chained super() calls in a dynamic context,
                    # where the dynamic type of the calling object is different
                    # from the static type of the class where the super() call
                    # site is - are never used by Pyan for resolving super().
                    #
                    # This is a limitation of pure lexical scope based static
                    # code analysis.
                    #
                    if len(self.mro[class_node]) > 1:
                        result = self.mro[class_node][1]
                        self.logger.debug("super of %s is %s" % (class_node, result))
                        return result
                    else:
                        msg = "super called for %s, but no known bases" % (class_node)
                        self.logger.info(msg)
                        raise UnresolvedSuperCallError(msg)
                else:
                    msg = "super called for %s, but MRO not determined for it (maybe still in pass 1?)" % (class_node)
                    self.logger.info(msg)
                    raise UnresolvedSuperCallError(msg)

            if funcname in ("str", "repr"):
                if len(ast_node.args) == 1:  # these take only one argument
                    obj_astnode = ast_node.args[0]
                    if isinstance(obj_astnode, (ast.Name, ast.Attribute)):
                        self.logger.debug("Resolving %s() of %s" % (funcname, get_ast_node_name(obj_astnode)))
                        attrname = "__%s__" % (funcname)
                        # build a temporary ast.Attribute AST node so that we can use get_attribute()
                        tmp_astnode = ast.Attribute(value=obj_astnode, attr=attrname, ctx=obj_astnode.ctx)
                        obj_node, attr_node = self.get_attribute(tmp_astnode)
                        attr_node.set_type("str")
                        self.logger.debug(
                            "Resolve %s() of %s: returning attr node %s"
                            % (funcname, get_ast_node_name(obj_astnode), attr_node)
                        )
                        return attr_node

            # add implementations for other built-in funcnames here if needed

    def resolve_attribute(self, ast_node)-> (Node):
        """Resolve an ast.Attribute.

        Nested attributes (a.b.c) are automatically handled by recursion.

        Return attr_node, like Node(B.c), and set self.last_value to the
        corresponding type, like C.
        """

        if not isinstance(ast_node, ast.Attribute):
            raise TypeError("Expected ast.Attribute; got %s" % (type(ast_node)))

        self.logger.debug(
            "Resolve %s.%s in context %s" % (get_ast_node_name(ast_node.value), ast_node.attr, type(ast_node.ctx))
        )
        
        # look up attr_name in the given namespace, return Node or None
        def lookup(ns, attr_name):
            if ns in self.scopes:
                sc = self.scopes[ns]
                if attr_name in sc.defs:
                    return sc.defs[attr_name]
            return None

        # Resolve nested attributes
        #
        # In pseudocode, e.g. "a.b.c" is represented in the AST as:
        #    ast.Attribute(attr=c, value=ast.Attribute(attr=b, value=a))
        #
        from_node = self.get_node_of_current_namespace()
        
        obj_name = get_ast_node_name(ast_node.value)
        
        
        
        # 新定义另一个虚拟节点，表示当前attr的obj，类型为返回的last_value
        if isinstance(ast_node.value, ast.Attribute):
            obj_node, attr_node = self.resolve_attribute(ast_node.value)
            if isinstance(attr_node, Node):
                obj_node = attr_node
            else:
                obj_node = self.get_node("", obj_name, None, flavor=Flavor.UNKNOWN)
                obj_node.set_type(obj_name)
            

        elif isinstance(ast_node.value, (ast.Num, ast.Str)): 
            t = type(ast_node.value)
            tn = t.__name__
            obj_node = self.get_node("", tn, None, flavor=Flavor.CLASS)
            obj_node.set_type(tn)

        # attribute of a function call. Detect cases like super().dostuff()
        else:
            self.visit(ast_node.value)
            if isinstance(self.last_value, Node):
                obj_node = self.last_value
            elif obj_name:
                obj_node = self.get_node("", obj_name, None, flavor=Flavor.UNKNOWN)
            else:
                obj_node = self.get_node("", "UNKNOWN", None, flavor=Flavor.UNKNOWN)
            obj_node.set_type(self.last_type)

        # 新定义一个虚拟节点，表示当前的attr
        attr_node = self.get_node(obj_node.get_name(), ast_node.attr, None, flavor=Flavor.ATTRIBUTE)
        
        if attr_node.get_name() == 'ReText.dialogs.EncodingDialog.handleTextChanged.buttonBox':
            qika = 1
        
                
        # 根据obj_node.type找出对应attr的obj
        obj_ns, obj_name = self.split(obj_node.get_type())
        class_node = self.find_node(obj_ns, obj_name)
        if class_node:
            obj_node = class_node
            # first try directly in object's ns (this works already in pass 1)
            ns = class_node.get_name()
            value_node = lookup(ns, ast_node.attr)
            # next try ns of each ancestor (this works only in pass 2,
            # after self.mro has been populated)
            #
            if value_node is None and class_node in self.mro:
                base_node = None
                for up in self.mro[class_node]:  # the first element is always obj itself
                    obj_ns, obj_name = self.split(up.get_type())
                    base_node = self.find_node(obj_ns, obj_name)
                    if base_node is None:
                        qika = 1
                    if not isinstance(base_node, Node):
                        continue
                    ns = base_node.get_name()
                    value_node = lookup(ns, ast_node.attr)
                    if value_node is not None:
                        break
                class_node = base_node
                if base_node is None:
                    qika = 1
            if class_node:
                obj_node = class_node
        
        if obj_node is None:
            qika = 1
        
        self.logger.debug("Attr %s from %s" % (ast_node.attr, obj_node.get_type()))

        #如果obj_node是func，并且其中恰好定义了和attr同名的变量，则会错误链接到那个变量，因此如果obj_node是func，则直接断开链接并返回
        if obj_node and obj_node.flavor in [Flavor.CLASSMETHOD, Flavor.FUNCTION, Flavor.METHOD, Flavor.PROPERTYMETHOD, Flavor.STATICMETHOD]: 
            attr_node = self.get_node(obj_node.get_name() + "()", ast_node.attr, None, flavor=Flavor.UNKNOWN)
            return obj_node, attr_node
        
        #qika 分析出命名空间路径ns，另有attr对应的名称b，将ns.b加入调用关系
        used_node = self.get_node(obj_node.get_type(), ast_node.attr, None, flavor=Flavor.ATTRIBUTE)
        used_node.set_type(used_node.get_name()) if used_node.get_type() is None else None
        
        from_node = self.get_node_of_current_namespace()
        self.logger.debug("Use from %s to %s (use as a attribute)" % (from_node, used_node))
        if self.add_uses_edge(from_node, used_node, true_type=False):
            self.logger.info(
                "New edge added for Use from %s to %s (use as a attribute)" % (from_node, used_node)
            ) 
        if used_node.get_type() is None:
            attr_node.set_type(used_node.get_name())
        else:
            attr_node.set_type(used_node.get_type())
        
        # 如果attr是一个call并且有返回值，将attr_node的value设为返回值对应节点，type设为返回值类型
        # 这条是因为python中有些类函数A.b()可以直接写成A.b来调用
        inner_ns = attr_node.get_name()
        if inner_ns in self.scopes:
            return_sc = self.scopes[inner_ns]
            if return_sc.Return is not None:
                namespace, name = self.split(return_sc.Return)
                return_node = self.get_node(namespace, name)
                self.logger.debug("Replace call %s as it return %s" % (attr_node, return_node))
                attr_node.set_value(return_node)
                attr_node.set_type(return_node.get_type())
        

        return obj_node, attr_node

    ###########################################################################
    # Scope analysis

    def analyze_scopes(self, code, filename):
        """Gather lexical scope information."""

        # Below, ns is the fully qualified ("dotted") name of sc.
        #
        # Technically, the module scope is anonymous, but we treat it as if
        # it was in a namespace named after the module, to support analysis
        # of several files as a set (keeping their module-level definitions
        # in different scopes, as we should).
        #
        scopes = {}

        def process(parent_ns, table):
            sc = Scope(table)
            ns = "%s.%s" % (parent_ns, sc.name) if len(sc.name) else parent_ns
            sc.path = ns
            scopes[ns] = sc
            for t in table.get_children():
                process(ns, t)

        process(self.module_name, symtable.symtable(code, filename, compile_type="exec"))

        # add to existing scopes (while not overwriting any existing definitions with None)
        for ns in scopes:
            if ns not in self.scopes:  # add new scope info
                self.scopes[ns] = scopes[ns]
            else:  # update existing scope info
                sc = scopes[ns]
                oldsc = self.scopes[ns]
                for name in sc.defs:
                    if name not in oldsc.defs:
                        oldsc.defs[name] = sc.defs[name]

        self.logger.debug("Scopes now: %s" % (self.scopes))

    def get_current_class(self):
        """Return the node representing the current class, or None if not inside a class definition."""
        return self.class_stack[-1] if len(self.class_stack) else None

    def get_node_of_current_namespace(self):
        """Return the unique node representing the current namespace,
        based on self.name_stack.

        For a Node n representing a namespace:
          - n.namespace = fully qualified name of the parent namespace
                          (empty string if at top level)
          - n.name      = name of this namespace
          - no associated AST node.
        """
        if len(self.name_stack) == 0:
            return None
        # name_stack should never be empty (always at least module name)

        namespace = ".".join(self.name_stack[0:-1])
        name = self.name_stack[-1]
        return self.get_node(namespace, name, None, flavor=Flavor.NAMESPACE)

    ###########################################################################
    # Value getter and setter

    def get_value(self, name):
        """Get the value of name in the current scope. Return the Node, or None if name is not set to a value."""

        # get the innermost scope that has name **and where name has a value**
        def find_scope(name):
            for sc in reversed(self.scope_stack):
                if name in sc.defs and sc.defs[name] is not None:
                    return sc
        sc = find_scope(name)
        if sc is not None:
            name_node = sc.defs[name]
            if isinstance(name_node, Node):
                self.logger.info("Get %s in %s, found in %s, type %s" % (name, self.scope_stack[-1], sc, name_node.get_type()))
                return name_node, sc.path
            else:
                # TODO: should always be a Node or None
                self.logger.debug(
                    "Get %s in %s, found in %s: not define" % (name, self.scope_stack[-1], sc)
                )
                return None, sc.path
        else:
            self.logger.debug("Get %s in %s: no Node value (or name not in scope)" % (name, self.scope_stack[-1]))
        return None, None
        
    def set_value(self, name, new_value=None, new_type=None, defined=True, node = None):
        """Set the value of name in the current scope. Value must be a Node."""

        # get the innermost scope that has name (should be the current scope unless name is a global)
        def find_scope(name):
            for sc in reversed(self.scope_stack):
                if name in sc.defs:
                    return sc

        sc = find_scope(name)
        if sc is not None:
            if (name not in sc.defs or sc.defs[name] is None or sc.defs[name].get_name().find("^^^argument^^^") != -1):
                sc.defs[name] = self.get_node(sc.path, name, node, flavor=Flavor.NAME)
            sc.defs[name].defined = defined
            if isinstance(new_value, Node):
                sc.defs[name].set_type(new_value.get_type())
                sc.defs[name].set_value(new_value)
                self.logger.info("Set %s in %s to %s" % (name, sc, new_value.get_type()))
            elif new_type is not None:
                sc.defs[name].set_type(new_type)
                old_value = sc.defs[name].get_value()
                if isinstance(old_value, Node) and old_value.get_type() != new_type:
                    sc.defs[name].set_value(None)
                    
                self.logger.debug("Set %s in %s to %s:%s" % (name, sc,sc.defs[name], new_type))
            else:
                # TODO: should always be a Node or None
                self.logger.debug("Set %s in %s false: type is None" % (name, sc))
                
        else:
            self.logger.debug("Set: name %s not in scope" % (name))

    ###########################################################################get_attribute
    # Attribute getter and setter

    def get_attribute(self, ast_node):
        """Get value of an ast.Attribute.

        Supports inherited attributes. If the obj's own namespace has no match
        for attr, the ancestors of obj are also tried, following the MRO based
        on the static type of the object, until one of them matches or until
        all ancestors are exhausted.

        Return pair of Node objects (obj,attr), where each item can be None
        on lookup failure. (Object not known, or no Node value assigned
        to its attr.)

        May pass through UnresolvedSuperCallError.
        
        qika：尝试判断attribute中调用者和调用值的类型,并将last_value设为actual_path
        
        当对a.b无法获取真实节点时，对应的actual_path设置为(a.actual_path).b，
        """

        if not isinstance(ast_node, ast.Attribute):
            raise TypeError("Expected ast.Attribute; got %s" % (type(ast_node)))
        if not isinstance(ast_node.ctx, ast.Load):
            raise ValueError("Expected a load context, got %s" % (type(ast_node.ctx)))

        obj_node, attr_node = self.resolve_attribute(ast_node)
        return obj_node, attr_node

        

    def set_attribute(self, ast_node, new_value, new_type):
        """Assign the Node provided as new_value into the attribute described
        by the AST node ast_node. Return True if assignment was done,
        False otherwise.

        May pass through UnresolvedSuperCallError.
        """

        if not isinstance(ast_node, ast.Attribute):
            raise TypeError("Expected ast.Attribute; got %s" % (type(ast_node)))
        if not isinstance(ast_node.ctx, ast.Store):
            raise ValueError("Expected a store context, got %s" % (type(ast_node.ctx)))

        obj_node, attr_node = self.resolve_attribute(ast_node)
        attr_name = attr_node.name
        
        if obj_node.get_name() == 'xmnlp.pinyin.pinyin.Pinyin':
            qika = 1
        

        if isinstance(obj_node, Node) and obj_node.namespace is not None:
            ns = obj_node.get_name()  # fully qualified namespace **of attr**
            if ns in self.scopes:
                sc = self.scopes[ns]
                
                if sc is not None:
                    if (attr_name not in sc.defs or sc.defs[attr_name] is None or sc.defs[attr_name].get_name().find("^^^argument^^^")):
                        sc.defs[attr_name] = self.get_node(ns, attr_name, None, flavor=Flavor.NAME)
                        sc.defs[attr_name].defined = True
                    if isinstance(new_value, Node) and new_type is not None:
                        attr_node.set_value(new_value)
                        attr_node.set_type(new_type)
                        
                        attr_node.defined = True
                        sc.defs[attr_name].defined = True

                        sc.defs[attr_name].set_value(attr_node.get_value())
                        sc.defs[attr_name].set_type(new_type)
                        self.logger.info("Set %s in %s to %s:%s" % (attr_name, sc, new_value, new_type))
                        return True
                    if new_type is not None:
                        new_value_ns, new_value_name = self.split(new_type)
                        new_value = self.get_node(new_value_ns, new_value_name, None, flavor=Flavor.UNKNOWN)
                        
                        attr_node.set_type(new_type)
                        attr_node.defined = True
                        sc.defs[attr_name].defined = True
                        
                        sc.defs[attr_name].set_type(new_type)
                        self.logger.info("Set %s in %s to Unknown:%s" % (attr_name, sc, new_type))
                    else:
                        # TODO: should always be a Node or None
                        self.logger.debug("Set %s in %s false: type is None" % (attr_name, sc))
                else:
                    self.logger.debug("Set: namespace %s not in scope" % (ns))
        return False

    ###########################################################################
    # Graph creation

    def get_node(self, namespace, name, ast_node=None, flavor=Flavor.UNSPECIFIED, actual_path = None, value = None, defined_path = None):
        """Return the unique node matching the namespace and name.
        Create a new node if one doesn't already exist.

        To associate the node with a syntax object in the analyzed source code,
        an AST node can be passed in. This only takes effect if a new Node
        is created.

        To associate an AST node to an existing graph node,
        see associate_node().

        Flavor describes the kind of object the node represents.
        See the node.Flavor enum for currently supported values.

        For existing nodes, flavor overwrites, if the given flavor is
        (strictly) more specific than the node's existing one.
        See node.Flavor.specificity().

        !!!
        In CallGraphVisitor, always use get_node() to create nodes, because it
        also sets some important auxiliary information. Do not call the Node
        constructor directly.
        !!!
        """
        if name == None:
            return None

        if name in self.nodes:
            for n in self.nodes[name]:
                if n.namespace == namespace:
                    if Flavor.specificity(flavor) > Flavor.specificity(n.flavor):
                        n.flavor = flavor
                    if n.ast_node is None and ast_node is not None:
                        n.ast_node = ast_node
                    return n

        # Try to figure out which source file this Node belongs to
        # (for annotated output).
        #
        # Other parts of the analyzer may change the filename later,
        # if a more authoritative source (e.g. a definition site) is found,
        # so the filenames should be trusted only after the analysis is
        # complete.
        #
        # TODO: this is tentative. Add in filename only when sure?
        # (E.g. in visit_ClassDef(), visit_FunctionDef())
        #
        if namespace in self.module_to_filename:
            # If the namespace is one of the modules being analyzed,
            # the the Node belongs to the correponding file.
            filename = self.module_to_filename[namespace]
        else:  # Assume the Node belongs to the current file.
            filename = self.filename

        n = Node(namespace, name, ast_node, filename, flavor, actual_path, value, defined_path)

        # Add to the list of nodes that have this short name.
        if name in self.nodes:
            self.nodes[name].append(n)
        else:
            self.nodes[name] = [n]

        return n
    
    def split(self, path: str)->[str]:
        """split path a.b.c.d as namespace a.b.c and name d
        """
        if path is None:
            return None, None
        name = path.split(".")[-1]
        namespace = path[0:-len(name)-1]
        if namespace == None:
            namespace = ""
        return namespace, name
        
        
    def find_node(self, namespace, name):
        """Return the unique node matching the namespace and name.
        Return None if not found.
        """
        if name in self.nodes:
            for n in self.nodes[name]:
                if n.namespace == namespace:
                    return n
        return None
    
    def find_scope_def_node(self, namespace, name):
        """Return the def value of name in scope namespce
        """
        if namespace in self.scopes:
            sc = self.scopes[namespace]
            if name in sc.defs and sc.defs[name] and sc.defs[name].defined:
                return sc.defs[name]
        return None
    
    def create_node(self, namespace, name, ast_node=None, flavor=Flavor.UNSPECIFIED, actual_path = None, value = None, defined_path = None):
        """
        Need some intermediate nodes to preserve specific call details, use this function instead of get_node() to ensure the creation of a new node. 
        Then can assign the actual node found by get_node() to the current node.
        """
        if namespace in self.module_to_filename:
            # If the namespace is one of the modules being analyzed,
            # the the Node belongs to the correponding file.
            filename = self.module_to_filename[namespace]
        else:  # Assume the Node belongs to the current file.
            filename = self.filename

        n = Node(namespace, name, ast_node, filename, flavor, actual_path, value, defined_path)

        # Add to the list of nodes that have this short name.
        if name in self.nodes:
            self.nodes[name].append(n)
        else:
            self.nodes[name] = [n]

        return n

    def get_parent_node(self, graph_node):
        """Get the parent node of the given Node. (Used in postprocessing.)"""
        if "." in graph_node.namespace:
            ns, name = graph_node.namespace.rsplit(".", 1)
        else:
            ns, name = "", graph_node.namespace
        return self.get_node(ns, name, None)

    def associate_node(self, graph_node, ast_node, filename=None):
        """Change the AST node (and optionally filename) mapping of a graph node.

        This is useful for generating annotated output with source filename
        and line number information.

        Sometimes a function in the analyzed code is first seen in a FromImport
        before its definition has been analyzed. The namespace can be deduced
        correctly already at that point, but the source line number information
        has to wait until the actual definition is found (because the line
        number is contained in the AST node). However, a graph Node must be
        created immediately when the function is first encountered, in order
        to have a Node that can act as a "uses" target (namespaced correctly,
        to avoid a wildcard and the over-reaching expand_unknowns() in cases
        where they are not needed).

        This method re-associates the given graph Node with a different
        AST node, which allows updating the context when the definition
        of a function or class is encountered."""
        graph_node.ast_node = ast_node
        if filename is not None:
            graph_node.filename = filename

    def add_defines_edge(self, from_node, to_node):
        """Add a defines edge in the graph between two nodes.
        N.B. This will mark both nodes as defined."""
        status = False
        from_node_name = from_node.get_name()
        if from_node_name not in self.defines_edges:
            self.defines_edges[from_node_name] = set()
            status = True
        from_node.defined = True
        if to_node is None or to_node in self.defines_edges[from_node_name]:
            return status
        self.defines_edges[from_node_name].add(to_node)
        to_node.defined = True
        return True

    
    def add_uses_edge(self, from_node, to_node, true_type = True):
        """Add a uses edge in the graph between two nodes."""
        if to_node is None:
            return False
        if from_node is None:
            return False
        
        from_scope = self.scopes[from_node.get_name()] if from_node.get_name() in self.scopes else None
        if from_scope is not None and to_node.name in from_scope.defs and from_scope.defs[to_node.name] is not None:
            if from_scope.defs[to_node.name].get_name() == to_node.get_name():
                self.logger.debug(
                    "%s is defined by %s, so skip" % (to_node, from_node)
                    ) 
                return False
    
        from_node_name = from_node.get_name()
        
        # true type为True时，表示use了某一个真实类型，可能是内置类型，自定义类或第三方引入类型等；
        # true type为False时，表示use了某一个具体实例，如外部自定义的变量a
        if true_type:
            # 获得被调用的真实节点
            to_node_path = to_node.get_name()
            to_node_type_path = to_node.get_type()

            while(to_node_type_path and to_node_type_path != to_node_path):
                type_name = to_node_type_path.split(".")[-1]
                type_namespace = to_node_type_path[0:-len(type_name)-1]
                type_node = self.find_node(type_namespace, type_name)
                if type_node == None:
                    break
                to_node = type_node
                to_node_path = to_node.get_name()
                to_node_type_path = to_node.get_type()
            
            if not to_node_type_path:
                to_node.set_type(to_node_path)
                to_node_type_path = to_node_path
            
            type_name = to_node_type_path.split(".")[-1]
            type_namespace = to_node_type_path[0:-len(type_name)-1]
            
            add_flag = False
            
            # 当type_namespace存在时，to_node即为真实节点
            if to_node_type_path in self.scopes:
                pass  
            elif (type_namespace in self.scopes and type_name in self.scopes[type_namespace].defs):
                to_node = self.scopes[type_namespace].defs[type_name]

            if to_node and to_node.defined == True:
                add_flag = True
                # 进入真实uses_edge添加逻辑
                if from_node_name not in self.uses_edges:
                    self.uses_edges[from_node_name] = set()
                if to_node in self.uses_edges[from_node_name]:
                    return False
                self.uses_edges[from_node_name].add(to_node)
                self.logger.debug(
                    "Added Use-Edge from %s to %s" % (from_node, to_node)
                    ) 
    
            # 当to_node_type_path和to_node_path不一致时，to_node为虚拟节点
            elif self.pas_time > 0 and not add_flag:
                # 当已经解析过一遍完整项目之后，进入虚拟uses_edge添加逻辑
                if from_node_name not in self.virtual_uses_edges:
                    self.virtual_uses_edges[from_node_name] = {}
                if to_node.namespace is None or (len(to_node.namespace) and '*' == to_node.namespace[0]) or '^^^argument^^^' in to_node.get_name():
                    to_node.namespace = 'UNKNOWN'
                if to_node in self.virtual_uses_edges[from_node_name]:
                    return False
                to_name = to_node.name
                if to_node.namespace in ["list", "dict", "set", "int", "float", "str", "bool", "tuple", "object", "NoneType"]:
                    return False
                
                self.virtual_uses_edges[from_node_name][to_node.get_name()]=set()
                
                if  to_node.flavor is Flavor.IMPORTEDITEM:
                    self.virtual_uses_edges[from_node_name][to_node.get_name()].add(to_node)
                
                prefix = ""
                name_list = to_node.namespace.split(".")
                for name in name_list:
                    prefix += name
                    prefix_node = self.find_node(prefix.rsplit(".",1)[0] if "." in prefix else "", name)
                    if prefix_node and prefix_node.flavor is Flavor.IMPORTEDITEM:
                        to_node.flavor = Flavor.IMPORTEDITEM
                        self.virtual_uses_edges[from_node_name][to_node.get_name()].add(to_node)
                        break
                    prefix += "."
                
                
                if to_name in self.nodes:
                    for n in self.nodes[to_name]:
                        # 添加可能的候选真实节点
                        if n.defined:
                            self.virtual_uses_edges[from_node_name][to_node.get_name()].add(n)
                            
                self.logger.debug(
                    "Added Virtual-Use-Edge from %s to %s" % (from_node, to_node)
                    ) 
            else:
                return False
        # true type为False时，表示use了某一个具体实例，如外部自定义的变量a
        else:
            """
            # 递归节点的值获取被定义的节点，即defined=True的节点
            while to_node.defined == False:
                if to_node.value == None:
                    break
                to_node = to_node.value
            """
            # 进入真实uses_edge添加逻辑
            if to_node.defined == True:
                if from_node_name not in self.uses_edges:
                    self.uses_edges[from_node_name] = set()
                if to_node in self.uses_edges[from_node_name]:
                    return False
                self.uses_edges[from_node_name].add(to_node)
                self.logger.debug(
                    "Added Use-Edge from %s to %s" % (from_node_name, to_node)
                    ) 
            # 当to_node_type_path和to_node_path不一致时，to_node为虚拟节点
            elif self.pas_time > 0:
                # 当已经解析过一遍完整项目之后，进入虚拟uses_edge添加逻辑
                if from_node_name not in self.virtual_uses_edges:
                    self.virtual_uses_edges[from_node_name] = {}
                if to_node.namespace is None or (len(to_node.namespace) and '*' == to_node.namespace[0]) or '^^^argument^^^' in to_node.get_name():
                    to_node.namespace = 'UNKNOWN'
                if to_node in self.virtual_uses_edges[from_node_name]:
                    return False
                to_name = to_node.name

                if to_node.namespace in ["list", "dict", "set", "int", "float", "str", "bool", "tuple", "object", "NoneType"]:
                    return False
                
                self.virtual_uses_edges[from_node_name][to_node.get_name()]=set()
                
                if to_node.flavor is Flavor.IMPORTEDITEM:
                    self.virtual_uses_edges[from_node_name][to_node.get_name()].add(to_node)
                
                prefix = ""
                name_list = to_node.namespace.split(".")
                for name in name_list:
                    prefix += name
                    prefix_node = self.find_node(prefix.rsplit(".",1)[0] if "." in prefix else "", name)
                    if prefix_node and prefix_node.flavor is Flavor.IMPORTEDITEM:
                        to_node.flavor = Flavor.IMPORTEDITEM
                        self.virtual_uses_edges[from_node_name][to_node.get_name()].add(to_node)
                        break
                    prefix += "."
                
                if to_name in self.nodes:
                    for n in self.nodes[to_name]:
                        # 添加可能的候选真实节点
                        if n.defined:
                            self.virtual_uses_edges[from_node_name][to_node.get_name()].add(n)
                            
                self.logger.debug(
                    "Added Virtual-Use-Edge from %s to %s" % (from_node, to_node)
                    ) 
            else:
                return False
                
            
        if isinstance(to_node, Node) and to_node.namespace is not None:
            self.remove_wild(from_node, to_node, to_node.name)

        return True
    

    def remove_wild(self, from_node, to_node, name):
        """Remove uses edge from from_node to wildcard *.name.

        This needs both to_node and name because in case of a bound name
        (e.g. attribute lookup) the name field of the *target value* does not
        necessarily match the formal name in the wildcard.

        Used for cleaning up forward-references once resolved.
        This prevents spurious edges due to expand_unknowns()."""

        if name is None:  # relative imports may create nodes with name=None.
            return

        if from_node not in self.uses_edges:  # no uses edges to remove
            return

        # Keep wildcard if the target is actually an unresolved argument
        # (see visit_FunctionDef())
        if to_node.get_name().find("^^^argument^^^") != -1:
            return

        # Here we may prefer to err in one of two ways:
        #
        #  a) A node seemingly referring to itself is actually referring
        #     to somewhere else that was not fully resolved, so don't remove
        #     the wildcard.
        #
        #     Example:
        #
        #         import sympy as sy
        #         def simplify(expr):
        #             sy.simplify(expr)
        #
        #     If the source file of sy.simplify is not included in the set of
        #     analyzed files, this will generate a reference to *.simplify,
        #     which is formally satisfied by this function itself.
        #
        #     (Actually, after commit e3c32b782a89b9eb225ef36d8557ebf172ff4ba5,
        #      this example is bad; sy.simplify will be recognized as an
        #      unknown attr of a known object, so no wildcard is generated.)
        #
        #  b) A node seemingly referring to itself is actually referring
        #     to itself (it can be e.g. a recursive function). Remove the wildcard.
        #
        #     Bad example:
        #
        #         def f(count):
        #             if count > 0:
        #                 return 1 + f(count-1)
        #             return 0
        #
        #     (This example is bad, because visit_FunctionDef() will pick up
        #      the f in the top-level namespace, so no reference to *.f
        #      should be generated in this particular case.)
        #
        # We choose a).
        #
        # TODO: do we need to change our opinion now that also recursive calls are visualized?
        #
        if to_node == from_node:
            return

        matching_wilds = [n for n in self.uses_edges[from_node] if n.namespace is None and n.name == name]
        assert len(matching_wilds) < 2  # the set can have only one wild of matching name
        if len(matching_wilds):
            wild_node = matching_wilds[0]
            self.logger.info("Use from %s to %s resolves %s; removing wildcard" % (from_node, to_node, wild_node))
            self.remove_uses_edge(from_node, wild_node)

    ###########################################################################
    # Postprocessing

    def contract_nonexistents(self):
        """For all use edges to non-existent (i.e. not defined nodes) X.name, replace with edge to *.name."""

        new_uses_edges = []
        removed_uses_edges = []
        for n in self.uses_edges:
            for n2 in self.uses_edges[n]:
                if n2.namespace is not None and not n2.defined:
                    n3 = self.get_node(None, n2.name, n2.ast_node)
                    n3.defined = False
                    new_uses_edges.append((n, n3))
                    removed_uses_edges.append((n, n2))
                    self.logger.info("Contracting non-existent from %s to %s as %s" % (n, n2, n3))

        for from_node, to_node in new_uses_edges:
            self.add_uses_edge(from_node, to_node)

        for from_node, to_node in removed_uses_edges:
            self.remove_uses_edge(from_node, to_node)

    def expand_unknowns(self):
        """For each unknown node *.name, replace all its incoming edges with edges to X.name for all possible Xs.

        Also mark all unknown nodes as not defined (so that they won't be visualized)."""

        new_defines_edges = []
        for n in self.defines_edges:
            for n2 in self.defines_edges[n]:
                if n2.namespace is None:
                    for n3 in self.nodes[n2.name]:
                        if n3.namespace is not None:
                            new_defines_edges.append((n, n3))

        for from_node, to_node in new_defines_edges:
            self.add_defines_edge(from_node, to_node)
            self.logger.info("Expanding unknowns: new defines edge from %s to %s" % (from_node, to_node))

        new_uses_edges = []
        for n in self.uses_edges:
            for n2 in self.uses_edges[n]:
                if n2.namespace is None:
                    for n3 in self.nodes[n2.name]:
                        if n3.namespace is not None:
                            new_uses_edges.append((n, n3))

        for from_node, to_node in new_uses_edges:
            self.add_uses_edge(from_node, to_node)
            self.logger.info("Expanding unknowns: new uses edge from %s to %s" % (from_node, to_node))

        for name in self.nodes:
            for n in self.nodes[name]:
                if n.namespace is None:
                    n.defined = False

    def cull_inherited(self):
        """
        For each use edge from W to X.name, if it also has an edge to W to Y.name where
        Y is used by X, then remove the first edge.
        """

        removed_uses_edges = []
        for n in self.uses_edges:
            for n2 in self.uses_edges[n]:
                inherited = False
                for n3 in self.uses_edges[n]:
                    if (
                        n3.name == n2.name
                        and n2.namespace is not None
                        and n3.namespace is not None
                        and n3.namespace != n2.namespace
                    ):
                        pn2 = self.get_parent_node(n2)
                        pn3 = self.get_parent_node(n3)
                        # if pn3 in self.uses_edges and pn2 in self.uses_edges[pn3]:
                        # remove the second edge W to Y.name (TODO: add an option to choose this)
                        if pn2 in self.uses_edges and pn3 in self.uses_edges[pn2]:  # remove the first edge W to X.name
                            inherited = True

                if inherited and n in self.uses_edges:
                    removed_uses_edges.append((n, n2))
                    self.logger.info("Removing inherited edge from %s to %s" % (n, n2))

        for from_node, to_node in removed_uses_edges:
            self.remove_uses_edge(from_node, to_node)

    def collapse_inner(self):
        """Combine lambda and comprehension Nodes with their parent Nodes to reduce visual noise.
        Also mark those original nodes as undefined, so that they won't be visualized."""

        # Lambdas and comprehensions do not define any names in the enclosing
        # scope, so we only need to treat the uses edges.

        # BUG: resolve relative imports causes (RuntimeError: dictionary changed size during iteration)
        # temporary solution is adding list to force a copy of 'self.nodes'
        for name in list(self.nodes):
            if name in ("lambda", "listcomp", "setcomp", "dictcomp", "genexpr"):
                if name in ["listcomp"]:
                    qika = 1
                for n in self.nodes[name]:
                    pn = self.get_parent_node(n)
                    if n.get_name() in self.uses_edges:
                        if n.get_name() == "asciimatics.effects.Matrix.reset.listcomp":
                            qika = 1
                        for n2 in self.uses_edges[n.get_name()]:  # outgoing uses edges
                            self.logger.info("Collapsing inner from %s to %s, uses %s" % (n, pn, n2))
                            if n2.defined:
                                self.add_uses_edge(pn, n2, true_type=False)
                            else:
                                self.add_uses_edge(pn, n2)
                    n.defined = False