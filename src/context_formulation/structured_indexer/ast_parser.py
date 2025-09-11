import ast
import os
import json
import logging
import builtins
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
import fnmatch
import re
import sys
import os

logger = logging.getLogger(__name__)

BUILTIN_TYPES = {name for name in dir(builtins)}
STANDARD_MODULES = {
    'abc', 'argparse', 'array', 'asyncio', 'base64', 'collections', 'copy', 
    'csv', 'datetime', 'enum', 'functools', 'glob', 'io', 'itertools', 
    'json', 'logging', 'math', 'os', 'pathlib', 'random', 're', 'shutil', 
    'string', 'sys', 'time', 'typing', 'uuid', 'warnings', 'xml'
}
EXCLUDED_NAMES = {'self', 'cls'}

@dataclass
class CodeComponent:
    """
    Represents a single code component (function, class, or variable) in a Python codebase.
    
    Stores the component's identifier, AST node, dependencies, and other metadata.
    """
    # Unique identifier for the component
    id: str
    
    # AST node representing this component
    node: ast.AST
    
    # Type of component
    component_type: str

    clean_name: str 
    
    # Full path to the file containing this component
    file_path: str
    
    # Relative path within the repo
    relative_path: str
    
    # Set of component IDs this component calls
    outgoing_calls: Dict[str, List[str]] = field(default_factory=lambda: {"class": [], "function": [], "variable": []})
    
    # Original source code of the component
    source_code: Optional[str] = None

    start_line: int = 0
    
    end_line: int = 0
    
    # Whether the component already has a docstring
    has_docstring: bool = False
    
    # Content of the docstring if it exists, empty string otherwise
    docstring: str = ""
    

    # External entities and same-file components not in outgoing_calls, categorized by type
    noise: Dict[str, List[str]] = field(default_factory=lambda: {"class": [], "function": [], "variable": []})

    # Total number of outgoing calls
    out_count: int = 0
    
    # Function/Class signature 
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert this component to a dictionary representation for JSON serialization."""
        return {
            'id': self.id,
            'component_type': self.component_type,
            'clean_name': self.clean_name,
            'file_path': self.file_path,
            'relative_path': self.relative_path,
            'outgoing_calls': self.outgoing_calls,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'has_docstring': self.has_docstring,
            'docstring': self.docstring,
            'source_code': self.source_code,
            'noise': self.noise,
            'out_count': self.out_count,
            'signature': self.signature
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CodeComponent':
        """Create a CodeComponent from a dictionary representation."""
        component = CodeComponent(
            id=data['id'],
            node=None,
            clean_name=data.get('clean_name', ""),
            component_type=data['component_type'],
            file_path=data['file_path'],
            relative_path=data['relative_path'],
            outgoing_calls=data.get('outgoing_calls', {"class": [], "function": [], "variable": []}),
            start_line=data.get('start_line', 0),
            end_line=data.get('end_line', 0),
            has_docstring=data.get('has_docstring', False),
            docstring=data.get('docstring', ""),
            source_code=data.get('source_code', ""),
            noise=data.get('noise', {"class": [], "function": [], "variable": []}),
            out_count=data.get('out_count', 0),
            signature=data.get('signature', "")
        )
        return component


def categorize_components_by_type(component_ids: List[str], all_components: Dict[str, 'CodeComponent']) -> Dict[str, List[str]]:
    """
    Categorize component IDs by their component type.
    
    Args:
        component_ids: List of component IDs to categorize.
        all_components: Dictionary of all components.
        
    Returns:
        Dict with keys 'class', 'function', 'variable' and lists of component IDs as values.
    """
    categorized = {"class": [], "function": [], "variable": []}
    
    for comp_id in component_ids:
        if comp_id in all_components:
            comp_type = all_components[comp_id].component_type
            if comp_type in ["method", "inner_function"]:
                continue
            if comp_type in categorized:
                categorized[comp_type].append(comp_id)
    
    return categorized


class ImportCollector(ast.NodeVisitor):
    """Collects import statements from Python code."""
    
    def __init__(self, current_file_path: str = "", repo_modules: Set[str] = None):
        self.current_file_path = current_file_path
        self.repo_modules = repo_modules or set()
        self.imports = set()
        self.from_imports = {}  
        self.import_aliases = {}  
        self.import_statements = []  
        self.original_names = {}  
        
    def visit_Import(self, node: ast.Import):
        """Process 'import x' statements."""
        import_names = []
        for name in node.names:
            if name.asname:
                import_names.append(f"{name.name} as {name.asname}")
            else:
                import_names.append(name.name)
        self.import_statements.append(f"import {', '.join(import_names)}")
        
        for name in node.names:
            module_path = name.name.replace(".", "/") + ".py"
            self.imports.add(module_path)
            if name.asname:
                self.import_aliases[name.asname] = module_path
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Process 'from x import y' statements."""
        if node.module is not None or node.level > 0:
            imported_names = []
            for name in node.names:
                if name.asname:
                    imported_names.append(f"{name.name} as {name.asname}")
                else:
                    imported_names.append(name.name)
            
            if node.level > 0:
                dots = '.' * node.level
                module_part = node.module if node.module else ''
                original_module = dots + module_part
            else:
                original_module = node.module
            
            self.import_statements.append(f"from {original_module} import {', '.join(imported_names)}")
            
            module = self._resolve_module_path(node.module, node.level)
            if module not in self.from_imports:
                self.from_imports[module] = []
            
            for name in node.names:
                if name.name == '*':
                    self.from_imports[module].append('*')
                else:
                    if name.asname:
                        self.from_imports[module].append(name.asname)
                        self.original_names[name.asname] = name.name
                        self.import_aliases[name.asname] = f"{module.replace('.py', '')}.{name.name}"
                    else:
                        imported_name = name.name
                        self.from_imports[module].append(imported_name)
        
        self.generic_visit(node)
    
    def _resolve_module_path(self, module_name: str, level: int = 0) -> str:
        if level > 0 and self.current_file_path:
            path_parts = self.current_file_path.split("/")
            dir_parts = path_parts[:-1]
            steps_up = level - 1
            if len(dir_parts) >= steps_up:
                target_dir_parts = dir_parts[:-steps_up] if steps_up > 0 else dir_parts
                
                if module_name:
                    module_parts = module_name.split(".")
                    target_parts = target_dir_parts + module_parts
                else:
                    target_parts = target_dir_parts
                
                resolved_path = "/".join(target_parts) + ".py"
                
                if resolved_path in self.repo_modules:
                    return resolved_path
        
        if module_name:
            module_path = module_name.replace(".", "/") + ".py"
            
            if level == 0 and self.current_file_path:
                current_dir = "/".join(self.current_file_path.split("/")[:-1])
                
                if current_dir:
                    potential_path = f"{current_dir}/{module_path}"
                    if potential_path in self.repo_modules:
                        return potential_path
            
            return module_path
        
        return ""


class ClassMethodDependencyCollector(ast.NodeVisitor):
    """Collects dependencies between classes through method calls."""
    
    def __init__(self, imports, from_imports, current_module, repo_modules, import_aliases):
        self.imports = imports
        self.from_imports = from_imports
        self.import_aliases = import_aliases
        self.current_module = current_module
        self.repo_modules = repo_modules
        self.class_dependencies = set()
        self.local_variables = set()
        
    def visit_Call(self, node: ast.Call):
        """Process method calls to identify class dependencies."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                obj_name = node.func.value.id
                method_name = node.func.attr
                
                if obj_name not in self.local_variables:
                    for module, imported_names in self.from_imports.items():
                        if obj_name in imported_names and module in self.repo_modules:
                            self.class_dependencies.add(f"{obj_name}@{module}")
                            break
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        """Track local variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_variables.add(target.id)
        self.generic_visit(node)


class VariableCollector(ast.NodeVisitor):
    """Collects top-level variable assignments from Python code."""
    
    def __init__(self, source: str):
        self.variables = {}  
        self.source_lines = source.split('\n')
        
    def visit_Assign(self, node: ast.Assign):
        """Process variable assignments at module level."""
        if hasattr(node, 'parent') and isinstance(node.parent, ast.Module):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id
                    start_line = node.lineno
                    end_line = getattr(node, 'end_lineno', node.lineno)
                    
                    if start_line <= len(self.source_lines):
                        if end_line == start_line:
                            source_code = self.source_lines[start_line - 1]
                        else:
                            source_code = '\n'.join(self.source_lines[start_line-1:end_line])
                    else:
                        source_code = ""
                    
                    self.variables[var_name] = (start_line, end_line, source_code)
        
        self.generic_visit(node)


class DependencyCollector(ast.NodeVisitor):
    """
    Collects dependencies between code components by analyzing
    attribute access, function calls, and class references.
    """
    
    def __init__(self, imports, from_imports, current_module, repo_modules, import_aliases, all_components=None):
        self.imports = imports
        self.from_imports = from_imports
        self.import_aliases = import_aliases  # alias -> original_name
        self.current_module = current_module
        self.repo_modules = repo_modules
        self.dependencies = set()
        self._current_class = None
        self.all_components = all_components or {}
        self.local_variables = set()
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Process class definitions."""
        old_class = self._current_class
        self._current_class = node.name
        
        for base in node.bases:
            if isinstance(base, ast.Name):
                self._add_dependency(base.id)
            elif isinstance(base, ast.Attribute):
                self._process_attribute(base)
        
        self.generic_visit(node)
        self._current_class = old_class
    
    def visit_Assign(self, node: ast.Assign):
        """Track local variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.local_variables.add(target.id)
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Process function calls."""
        if isinstance(node.func, ast.Name):
            self._add_dependency(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self._process_attribute(node.func, is_method_call=True)
        
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Process name references."""
        if isinstance(node.ctx, ast.Load):
            self._add_dependency(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Process attribute access."""
        self._process_attribute(node)
        self.generic_visit(node)
    
    def _process_attribute(self, node: ast.Attribute, is_method_call=False):
        """Process an attribute node to extract potential dependencies."""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.insert(0, current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.insert(0, current.id)
            
            if parts[0] in self.local_variables:
                return
                
            if parts[0] in EXCLUDED_NAMES:
                return
                
            if parts[0] in self.imports:
                module_path = parts[0]
                module_name = module_path.replace("/", ".").replace(".py", "")
                if module_name in STANDARD_MODULES:
                    return
                    
                if module_path in self.repo_modules:
                    if len(parts) > 1:
                        target_id = f"{parts[1]}@{module_path}"
                        if is_method_call and target_id in self.all_components:
                            if self.all_components[target_id].component_type == "method":
                                containing_class = self._find_containing_class(target_id)
                                if containing_class:
                                    self.dependencies.add(containing_class)
                                    return
                        self.dependencies.add(target_id)
            
            for module, imported_names in self.from_imports.items():
                module_name = module.replace("/", ".").replace(".py", "")
                if module_name in STANDARD_MODULES:
                    continue
                
                if '*' in imported_names and module in self.repo_modules:
                    target_id = f"{parts[0]}@{module}"
                    if target_id in self.all_components:
                        comp_type = self.all_components[target_id].component_type
                        if comp_type in ["class", "function", "variable"]:
                            if is_method_call and self.all_components[target_id].component_type == "method":
                                containing_class = self._find_containing_class(target_id)
                                if containing_class:
                                    self.dependencies.add(containing_class)
                                    return
                            self.dependencies.add(target_id)
                            return
                elif parts[0] in imported_names and module in self.repo_modules:
                    target_id = f"{parts[0]}@{module}"
                    if is_method_call and target_id in self.all_components:
                        if self.all_components[target_id].component_type == "method":
                            containing_class = self._find_containing_class(target_id)
                            if containing_class:
                                self.dependencies.add(containing_class)
                                return
                    self.dependencies.add(target_id)
                    return
    
    def _find_containing_class(self, method_id: str) -> Optional[str]:
        """Find the containing class for a method."""
        if "." in method_id.split("@")[0]:
            class_and_method = method_id.split("@")[0]
            class_name, method_name = class_and_method.split(".", 1)
            module_path = method_id.split("@")[1]
            return f"{class_name}@{module_path}"
        
        method_name = method_id.split("@")[0]
        module_path = method_id.split("@")[1]
        
        for comp_id, component in self.all_components.items():
            if (component.component_type == "class" and 
                comp_id.endswith(f"@{module_path}")):
                if hasattr(component, 'node') and component.node:
                    for method_node in component.node.body:
                        if (isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                            method_node.name == method_name):
                            return comp_id
        return None
    
    def _add_dependency(self, name):
        """Add a potential dependency based on a name reference."""
        if name in BUILTIN_TYPES:
            return
            
        if name in EXCLUDED_NAMES:
            return
            
        if name in self.local_variables:
            return
        
        if name in self.import_aliases:
            original = self.import_aliases[name]
            if "." in original and not original.endswith(".py"):
                module_parts = original.rsplit(".", 1)
                module = module_parts[0].replace(".", "/") + ".py"
                func_name = module_parts[1]
                if module in self.repo_modules:
                    target_id = f"{func_name}@{module}"
                    if target_id in self.all_components:
                        if self.all_components[target_id].component_type == "method":
                            containing_class = self._find_containing_class(target_id)
                            if containing_class:
                                self.dependencies.add(containing_class)
                                return
                    self.dependencies.add(target_id)
                    return
            else:
                if original in self.repo_modules:
                    self.dependencies.add(f"{name}@{original}")
                    return
            
        for module, imported_names in self.from_imports.items():
            module_name = module.replace("/", ".").replace(".py", "")
            if module_name in STANDARD_MODULES:
                continue
            
            if '*' in imported_names and module in self.repo_modules:
                target_id = f"{name}@{module}"
                if target_id in self.all_components:
                    comp_type = self.all_components[target_id].component_type
                    if comp_type in ["class", "function", "variable"]:
                        if self.all_components[target_id].component_type == "method":
                            containing_class = self._find_containing_class(target_id)
                            if containing_class:
                                self.dependencies.add(containing_class)
                                return
                        self.dependencies.add(target_id)
                        return
            elif name in imported_names and module in self.repo_modules:
                target_id = f"{name}@{module}"
                if target_id in self.all_components:
                    if self.all_components[target_id].component_type == "method":
                        containing_class = self._find_containing_class(target_id)
                        if containing_class:
                            self.dependencies.add(containing_class)
                            return
                self.dependencies.add(target_id)
                return
                
        local_component_id = f"{name}@{self.current_module}"
        if local_component_id in self.all_components:
            if self.all_components[local_component_id].component_type == "method":
                containing_class = self._find_containing_class(local_component_id)
                if containing_class:
                    self.dependencies.add(containing_class)
                    return
        self.dependencies.add(local_component_id)


def add_parent_to_nodes(tree: ast.AST) -> None:
    """
    Add a 'parent' attribute to each node in the AST.
    
    Args:
        tree: The AST to process
    """
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node


class DependencyParser:
    """
    Parses Python code to build a dependency graph between code components.
    """
    
    def __init__(self, repo_path: str):
        """
        Initialize the dependency parser.
        
        Args:
            repo_path: Path to the Python code repository.
        """
        self.repo_path = os.path.abspath(repo_path)
        self.components: Dict[str, CodeComponent] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.modules: Set[str] = set()
        self.external_knowledge: Dict[str, Dict[str, List[str]]] = {}
        
    def parse_repository(self):
        """
        Parse all Python files in the repository to build the dependency graph.
        
        Returns:
            Dict[str, CodeComponent]: Dictionary of parsed components.
        """
        logger.info(f"Parsing repository at {self.repo_path}")
        
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                
                module_path = self._file_to_module_path(relative_path)
                self.modules.add(module_path)
                
                self._parse_file(file_path, relative_path, module_path)
        
        self._resolve_dependencies()

        self._extract_external_knowledge()

        self._add_noise_to_components()

        self._update_out_counts()

        logger.info(f"Found {len(self.components)} code components")
        return self.components
    
    def _file_to_module_path(self, file_path: str) -> str:
        """
        Convert a file path to a Python module path.
        
        Args:
            file_path: The file path to convert.
            
        Returns:
            str: The module path.
        """
        return file_path
    
    def _parse_file(self, file_path: str, relative_path: str, module_path: str):
        """
        Parse a single Python file to collect code components.
        
        Args:
            file_path: Absolute path to the Python file.
            relative_path: Relative path within the repository.
            module_path: Module path for the file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            add_parent_to_nodes(tree)
            
            import_collector = ImportCollector(relative_path, self.modules)
            import_collector.visit(tree)

            variable_collector = VariableCollector(source)
            variable_collector.visit(tree)

            self._collect_components(tree, file_path, relative_path, module_path, source, variable_collector.variables)
            
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.warning(f"Error parsing {file_path}: {e}")
    
    def _collect_components(self, tree: ast.AST, file_path: str, relative_path: str, 
                          module_path: str, source: str, variables: Dict[str, Tuple[int, int, str]]):
        """
        Collect all code components (functions, classes, variables) from an AST.
        
        Args:
            tree: The AST tree to analyze.
            file_path: Absolute path to the Python file.
            relative_path: Relative path within the repository.
            module_path: Module path for the file.
            source: Source code of the file.
            variables: Dictionary of variables found in the file.
        """
        for var_name, (start_line, end_line, var_source_code) in variables.items():
            var_id = f"{var_name}@{module_path}"
            
            component = CodeComponent(
                id=var_id,
                node=None,
                clean_name=var_name,
                component_type="variable",
                file_path=file_path,
                relative_path=relative_path,
                source_code=var_source_code,
                start_line=start_line,
                end_line=end_line,
                has_docstring=False,
                docstring=""
            )
            
            self.components[var_id] = component
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_id = f"{node.name}@{module_path}"
                
                has_docstring = (
                    len(node.body) > 0 
                    and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                
                docstring = self._get_docstring(source, node) if has_docstring else "DOCSTRING"
                
                signature = self._get_class_signature(node, source)
                
                component = CodeComponent(
                    id=class_id,
                    node=node,
                    clean_name=node.name,
                    component_type="class",
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_source_segment(source, node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    has_docstring=has_docstring,
                    docstring=docstring,
                    signature=signature
                )
                
                self.components[class_id] = component
                
                for method_node in node.body:
                    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_id = f"{node.name}.{method_node.name}@{module_path}"
                        
                        method_has_docstring = (
                            len(method_node.body) > 0 
                            and isinstance(method_node.body[0], ast.Expr) 
                            and isinstance(method_node.body[0].value, ast.Constant)
                            and isinstance(method_node.body[0].value.value, str)
                        )
                        
                        method_docstring = self._get_docstring(source, method_node) if method_has_docstring else "DOCSTRING"
                        
                        method_signature = self._get_function_signature(method_node, source)
                        
                        method_component = CodeComponent(
                            id=method_id,
                            node=method_node,
                            clean_name=method_node.name,
                            component_type="method",
                            file_path=file_path,
                            relative_path=relative_path,
                            source_code=self._get_source_segment(source, method_node),
                            start_line=method_node.lineno,
                            end_line=getattr(method_node, "end_lineno", method_node.lineno),
                            has_docstring=method_has_docstring,
                            docstring=method_docstring,
                            signature=method_signature
                        )
                        
                        self.components[method_id] = method_component
        

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef):
                    continue
                
                parent_function = None
                if hasattr(node, 'parent'):
                    current = node.parent
                    while current:
                        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            parent_function = current
                            break
                        current = getattr(current, 'parent', None)

                if parent_function:
                    func_id = f"{parent_function.name}.{node.name}@{module_path}"
                    component_type = "inner_function"
                else:
                    func_id = f"{node.name}@{module_path}"
                    component_type = "function"
                
                has_docstring = (
                    len(node.body) > 0 
                    and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)
                )
                
                docstring = self._get_docstring(source, node) if has_docstring else "DOCSTRING"
                
                signature = self._get_function_signature(node, source)
                
                component = CodeComponent(
                    id=func_id,
                    node=node,
                    clean_name=node.name,
                    component_type=component_type,
                    file_path=file_path,
                    relative_path=relative_path,
                    source_code=self._get_source_segment(source, node),
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    has_docstring=has_docstring,
                    docstring=docstring,
                    signature=signature
                )
                
                self.components[func_id] = component
    
    def _get_parent_components(self, component_id: str, tree: ast.AST) -> Set[str]:
        """
        Get parent component IDs to prevent data leakage.
        
        Args:
            component_id: ID of the component to find parents for.
            tree: AST tree.
            
        Returns:
            Set of parent component IDs.
        """
        parent_components = set()
        module_path = component_id.split("@")[1]
        
        if "." in component_id.split("@")[0]:
            name_part = component_id.split("@")[0]
            if name_part.count(".") == 1:  
                parent_name, component_name = name_part.split(".", 1)                
                parent_class_id = f"{parent_name}@{module_path}"
                parent_function_id = f"{parent_name}@{module_path}"
                
                if parent_class_id in self.components and self.components[parent_class_id].component_type == "class":
                    parent_components.add(parent_class_id)
                elif parent_function_id in self.components and self.components[parent_function_id].component_type == "function":
                    parent_components.add(parent_function_id)
            else:
                parts = name_part.split(".")
                for i in range(len(parts) - 1):
                    parent_part = ".".join(parts[:i+1])
                    parent_id = f"{parent_part}@{module_path}"
                    if parent_id in self.components:
                        parent_components.add(parent_id)
        else:
            component_name = component_id.split("@")[0]
        
        if "." not in component_id.split("@")[0]:
            component_name = component_id.split("@")[0]
            
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == component_name:
                        target_node = node
                        break
            
            if target_node:
                current = target_node
                while hasattr(current, 'parent'):
                    parent = current.parent
                    if isinstance(parent, ast.ClassDef):
                        parent_id = f"{parent.name}@{module_path}"
                        parent_components.add(parent_id)
                    elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        parent_id = f"{parent.name}@{module_path}"
                        parent_components.add(parent_id)
                    current = parent
        
        return parent_components
    
    def _resolve_dependencies(self):
        """
        Second pass to resolve dependencies between components.
        """
        for component_id, component in self.components.items():
            file_path = component.file_path
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                
                tree = ast.parse(source)
                add_parent_to_nodes(tree)
                import_collector = ImportCollector(component.relative_path, self.modules)
                import_collector.visit(tree)
                component_node = None
                module_path = self._file_to_module_path(component.relative_path)
                
                if "." in component_id.split("@")[0]:
                    name_part = component_id.split("@")[0]
                    parent_name, component_name = name_part.split(".", 1)
                else:
                    component_name = component_id.split("@")[0]
                
                parent_components = self._get_parent_components(component_id, tree)
                
                if component.component_type == "function":
                    for node in ast.walk(tree):
                        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                                and node.name == component_name):
                            component_node = node
                            break
                
                elif component.component_type == "inner_function":
                    for node in ast.walk(tree):
                        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                                and node.name == component_name
                                and hasattr(node, 'parent') and isinstance(node.parent, (ast.FunctionDef, ast.AsyncFunctionDef))):
                            component_node = node
                            break
                
                elif component.component_type == "method":
                    for node in ast.walk(tree):
                        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                                and node.name == component_name
                                and hasattr(node, 'parent') and isinstance(node.parent, ast.ClassDef)):
                            component_node = node
                            break
                
                elif component.component_type == "class":
                    for node in ast.iter_child_nodes(tree):
                        if isinstance(node, ast.ClassDef) and node.name == component_name:
                            component_node = node
                            break
                
                elif component.component_type == "variable":
                    continue
                
                if component_node:
                    dependency_collector = DependencyCollector(
                        import_collector.imports,
                        import_collector.from_imports,
                        module_path,
                        self.modules,
                        import_collector.import_aliases,
                        self.components  
                    )
                    
                    if isinstance(component_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        for arg in component_node.args.args:
                            dependency_collector.local_variables.add(arg.arg)
                            
                    dependency_collector.visit(component_node)
                    
                    if isinstance(component_node, ast.ClassDef):
                        class_method_collector = ClassMethodDependencyCollector(
                            import_collector.imports,
                            import_collector.from_imports,
                            module_path,
                            self.modules,
                            import_collector.import_aliases
                        )
                        class_method_collector.visit(component_node)
                        dependency_collector.dependencies.update(class_method_collector.class_dependencies)
                    
                    filtered_dependencies = set()
                    for dep in dependency_collector.dependencies:
                        if dep in parent_components:
                            continue
                            
                        if dep in self.components:
                            filtered_dependencies.add(dep)
                        elif "@" in dep:
                            if dep in self.components:
                                filtered_dependencies.add(dep)
                            else:
                                dep_name, dep_module = dep.split("@", 1)
                                if dep_module in self.modules:
                                    for comp_id in self.components:
                                        if comp_id.startswith(f"{dep_name}@{dep_module}"):
                                            if comp_id not in parent_components:
                                                filtered_dependencies.add(comp_id)
                                            break                    
                    filtered_for_outgoing = []
                    for dep in filtered_dependencies:
                        if dep in self.components:
                            dep_type = self.components[dep].component_type
                            if dep_type in ["class", "function", "variable"]:
                                filtered_for_outgoing.append(dep)
                    
                    categorized_dependencies = categorize_components_by_type(filtered_for_outgoing, self.components)
                    component.outgoing_calls = categorized_dependencies
                    component.out_count = sum(len(v) for v in categorized_dependencies.values())
                
            except (SyntaxError, UnicodeDecodeError) as e:
                logger.warning(f"Error analyzing dependencies in {file_path}: {e}")
                continue
    
    def _find_entity_source(self, package_init_path: str, entity_name: str) -> Optional[str]:
        """
        Find the actual source file where an entity is defined within a package.
        """
        package_dir = package_init_path.replace("/__init__.py", "")
        
        for module_path in self.modules:
            if module_path.startswith(package_dir + "/") and module_path.endswith(".py") and not module_path.endswith("__init__.py"):
                entity_id = f"{entity_name}@{module_path}"
                if entity_id in self.components:
                    return module_path
        
        return None
    
    def _extract_external_knowledge(self):
        """
        Extract external knowledge from import statements in each file.
        """
        for root, _, files in os.walk(self.repo_path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, self.repo_path)
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()    
                    tree = ast.parse(source)                    
                    import_collector = ImportCollector(relative_path, self.modules)
                    import_collector.visit(tree)                    
                    external_entities = []
                    
                    for module_path in import_collector.imports:
                        if module_path in self.modules:
                            for comp_id in self.components:
                                if comp_id.endswith(f"@{module_path}"):
                                    comp_type = self.components[comp_id].component_type
                                    if comp_type in ["class", "function", "variable"]:
                                        external_entities.append(comp_id)
                    
                    for module_path, imported_names in import_collector.from_imports.items():
                        if module_path in self.modules:
                            if '*' in imported_names:
                                for comp_id in self.components:
                                    if comp_id.endswith(f"@{module_path}"):
                                        comp_type = self.components[comp_id].component_type
                                        if comp_type in ["class", "function", "variable"]:
                                            external_entities.append(comp_id)
                            else:
                                for name in imported_names:
                                    if name != '*':  
                                        original_name = import_collector.original_names.get(name, name)
                                        entity_id = f"{original_name}@{module_path}"
                                        if entity_id in self.components:
                                            external_entities.append(entity_id)
                        else:
                            package_init_path = module_path.replace(".py", "") + "/__init__.py"
                            if package_init_path in self.modules:
                                for name in imported_names:
                                    if name != '*':
                                        actual_source = self._find_entity_source(package_init_path, name)
                                        if actual_source:
                                            original_name = import_collector.original_names.get(name, name)
                                            external_entities.append(f"{original_name}@{actual_source}")
                    
                    self.external_knowledge[relative_path] = {
                        'import_statements': import_collector.import_statements,
                        'external_entity': external_entities
                    }
                    
                except (SyntaxError, UnicodeDecodeError) as e:
                    logger.warning(f"Error extracting external knowledge from {file_path}: {e}")
                    continue
    
    def _add_noise_to_components(self):
        """
        Add noise field to each component based on external entities and same-file components not in outgoing_calls.
        """
        for component_id, component in self.components.items():
            relative_path = component.relative_path
            
            try:
                with open(component.file_path, "r", encoding="utf-8") as f:
                    source = f.read()
                tree = ast.parse(source)
                add_parent_to_nodes(tree)
                parent_components = self._get_parent_components(component_id, tree)
            except Exception:
                parent_components = set()
            
            noise_components = []
            
            if relative_path in self.external_knowledge:
                external_entities = self.external_knowledge[relative_path]['external_entity']
                noise_components.extend(external_entities)
            
            all_outgoing_calls = []
            for comp_type in component.outgoing_calls:
                all_outgoing_calls.extend(component.outgoing_calls[comp_type])
            
            same_file_components = [
                comp_id for comp_id, comp in self.components.items()
                if comp.relative_path == relative_path and comp_id != component_id
            ]
            noise_components.extend(same_file_components)
            
            noise_entities = []
            for entity in noise_components:
                if (entity not in all_outgoing_calls and 
                    entity not in parent_components and
                    entity in self.components):
                    entity_type = self.components[entity].component_type
                    if entity_type in ["class", "function"]:
                        noise_entities.append(entity)
            
            component.noise = categorize_components_by_type(noise_entities, self.components)
    
    def _update_out_counts(self):
        """
        Update out_count for all components to ensure consistency.
        """
        for component in self.components.values():
            component.out_count = sum(len(v) for v in component.outgoing_calls.values())
    
    def _get_source_segment(self, source: str, node: ast.AST) -> str:
        """
        Get source code segment for an AST node.
        
        Args:
            source: The source code of the file.
            node: The AST node to extract source for.
            
        Returns:
            str: The source code segment.
        """
        try:
            if hasattr(ast, "get_source_segment"):
                segment = ast.get_source_segment(source, node)
                if segment is not None:
                    return segment
            
            lines = source.split("\n")
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", node.lineno) - 1
            return "\n".join(lines[start_line:end_line + 1])
        
        except Exception as e:
            logger.warning(f"Error getting source segment: {e}")
            return ""
    
    def _get_docstring(self, source: str, node: ast.AST) -> str:
        """
        Get the docstring for a given AST node.
        
        Args:
            source: The source code of the file.
            node: The AST node to extract docstring from.
            
        Returns:
            str: The docstring content or empty string if none found.
        """
        try:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if (len(node.body) > 0 
                    and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                    return node.body[0].value.value
            return "DOCSTRING"
        except Exception as e:
            logger.warning(f"Error getting docstring: {e}")
            return "DOCSTRING"
    
    def _get_function_signature(self, node: ast.FunctionDef, source: str) -> str:
        """
        Extract function signature from AST node.
        
        Args:
            node: Function AST node.
            source: Source code of the file.
            
        Returns:
            str: Function signature.
        """
        try:
            lines = source.split('\n')
            start_line = node.lineno - 1
            
            signature_lines = []
            paren_count = 0
            found_opening = False
            
            for i in range(start_line, len(lines)):
                line = lines[i].strip()
                if not found_opening and 'def ' in line:
                    found_opening = True
                
                if found_opening:
                    signature_lines.append(lines[i])
                    paren_count += line.count('(') - line.count(')')
                    
                    if paren_count == 0 and ':' in line:
                        break
            
            signature = ' '.join(signature_lines).strip()
            
            import re
            signature = re.sub(r'\s+', ' ', signature)
            if not signature.endswith(':'):
                signature += ':'
                
            return signature
        except Exception as e:
            logger.warning(f"Error extracting function signature: {e}")
            return f"def {node.name}():"
    
    def _get_class_signature(self, node: ast.ClassDef, source: str) -> str:
        """
        Extract class signature from AST node.
        
        Args:
            node: Class AST node.
            source: Source code of the file.
            
        Returns:
            str: Class signature.
        """
        try:
            lines = source.split('\n')
            start_line = node.lineno - 1
            
            signature_lines = []
            paren_count = 0
            found_opening = False
            
            for i in range(start_line, len(lines)):
                line = lines[i].strip()
                if not found_opening and 'class ' in line:
                    found_opening = True
                
                if found_opening:
                    signature_lines.append(lines[i])
                    paren_count += line.count('(') - line.count(')')
                    
                    if ':' in line and (paren_count == 0 or '(' not in line):
                        break
            
            signature = ' '.join(signature_lines).strip()
            
            import re
            signature = re.sub(r'\s+', ' ', signature)
            if not signature.endswith(':'):
                signature += ':'
                
            return signature
        except Exception as e:
            logger.warning(f"Error extracting class signature: {e}")
            return f"class {node.name}:"

    def save_dependency_graph(self, output_path: str):
        """
        Save the dependency graph to a JSON file.
        
        Args:
            output_path: Path where to save the dependency graph.
        """
        serializable_components = {
            comp_id: component.to_dict()
            for comp_id, component in self.components.items()
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_components, f, indent=2)
        
        logger.info(f"Saved dependency graph to {output_path}")
    
    def save_external_knowledges(self, output_path: str):
        """
        Save the external knowledge to a JSON file.
        
        Args:
            output_path: Path where to save the external knowledge.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.external_knowledge, f, indent=2)
        
        logger.info(f"Saved external knowledge to {output_path}")
    
    def load_dependency_graph(self, input_path: str):
        """
        Load the dependency graph from a JSON file.
        
        Args:
            input_path: Path to load the dependency graph from.
            
        Returns:
            Dict[str, CodeComponent]: Dictionary of loaded components.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            serialized_components = json.load(f)
        
        self.components = {
            comp_id: CodeComponent.from_dict(comp_data)
            for comp_id, comp_data in serialized_components.items()
        }
        
        logger.info(f"Loaded {len(self.components)} components from {input_path}")
        return self.components
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse Python code to build a dependency graph.")
    parser.add_argument("--repo_path", type=str, default=None, help="Path to the Python code repository")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for dependency graph and external knowledge files")
    args = parser.parse_args()
    
    dependency_parser = DependencyParser(args.repo_path)
    components = dependency_parser.parse_repository()
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        repo_name = os.path.basename(os.path.abspath(args.repo_path))
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "parser_output", repo_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    dependency_graph_path = os.path.join(output_dir, "dependency_graph.json")
    external_knowledge_path = os.path.join(output_dir, "external_knowledge.json")
    
    dependency_parser.save_dependency_graph(dependency_graph_path)
    print(f"Dependency graph saved to: {dependency_graph_path}")
    
    dependency_parser.save_external_knowledges(external_knowledge_path)
    print(f"External knowledge saved to: {external_knowledge_path}")