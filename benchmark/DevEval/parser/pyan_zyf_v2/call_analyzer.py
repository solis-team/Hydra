

"""
    根据analyzer.py中的CallGraphVisitor类分析出的结果，构造object各个function间的调用关系
    输出组织成和object原文件夹相同的结构，每个原py文件对应位置放置一个同名的json文件，文件内容为该py文件中的function的调用信息
    调用信息包含以下内容：
        该function的名称
        该function的路径
        该function使用的import语句
        该function调用的其他function路径（按 In-Class，In-File，In-Object 三个层级分组）
"""

import os
import json
import dill as pickle


def get_file_name(path, object_path):
    
    file_path = ''
    path_list = path.split('.')
    for level, node in enumerate(path_list):
        for root, dirs, files in os.walk(object_path):
            for file in files:
                if file == node+'.py' and root == os.path.join([object_path] + path_list[:level]):
                    file_path = '.'.join(path_list[:level+1])
    return file_path
        

class CallAnalyzer(object):
    def __init__(self, nodes=None, define_edges=None, used_edges=None, virtual_used_edges=None, import_used_edges=None):
        self.nodes = nodes or []
        self.define_edges = define_edges or {}
        self.used_edges = used_edges or {}
        self.virtual_used_edges = virtual_used_edges or {}
        self.import_used_edges = import_used_edges or {}
         
    
    @classmethod
    def from_visitor(cls, visitor, object_path, prefix=None, options=None, logger=None):
        """
        prefix: 形如 "A.B.C" 的字符串，表示需要分析的函数路径的前缀
        """
        # collect and sort defined nodes
        visited_nodes = {}
        for name in visitor.nodes:
            for node in visitor.nodes[name]:
                if node.defined and node.namespace is not None:
                    if prefix == None or prefix == node.get_name()[:len(prefix)]:
                        visited_nodes[node.get_name()] = node
        # visited_nodes.sort(key=lambda x: (x[0]))
        
        # Now add define edges
        define_edges = {}
        for n in visitor.defines_edges:
            user_node = visited_nodes[n] if n in visited_nodes else None
            define_edges[user_node] = []
            for n2 in visitor.defines_edges[n]:
                define_edges[user_node].append(n2)
        
        # Now add call edges
        used_edges = {}
        for n in visitor.uses_edges:
            user_node = visited_nodes[n] if n in visited_nodes else None
            if n == 'asyncssh.sftp.SFTPClientHandler.statvfs':
                qika = 1
            if user_node and user_node.defined and user_node.namespace is not None:
                caller_file = user_node.filename
                caller_type = user_node.flavor.value
                caller_flag = False
                used_edges[user_node] = {"in_class": [], "in_file": [], "in_object": []}
                
                if caller_type in ["method","staticmethod","classmethod","propertymethod"]:
                    caller_class = user_node.namespace.split(".")[-1]
                elif caller_type == "function":
                    caller_class = None
                elif caller_type == "class":
                    caller_class = user_node.name
                elif caller_type == "module":
                    caller_class = None
                else:
                    caller_class = user_node.namespace.split(".")[-1]
                    #raise Exception("Unknown caller type: %s" % caller_type)
                    
                for n2 in visitor.uses_edges[n]:
                    if n2.namespace is not None and n2.namespace != "*":
                        caller_flag = True
                        callee_file = n2.filename
                        callee_type = n2.flavor.value
                        if callee_type in ["method","staticmethod","classmethod","propertymethod","attribute"]:
                            callee_class = n2.namespace.split(".")[-1]
                        elif callee_type == "function":
                            callee_class = None
                        elif callee_type == "class":
                            callee_class = n2.name
                        elif callee_type == "module":
                            callee_class = None
                        else:
                            callee_class = n2.namespace.split(".")[-1]
                            # raise Exception("Unknown callee type: %s" % callee_type)
                        
                        if callee_file == caller_file and n in n2.namespace:
                            pass
                        elif callee_class == caller_class and callee_class is not None:
                            used_edges[user_node]["in_class"].append(n2)
                        elif callee_file == caller_file:
                            if len(callee_file.split('.')) > 2:
                                qika = 1
                            used_edges[user_node]["in_file"].append(n2)
                        else:
                            used_edges[user_node]["in_object"].append(n2)
                            
                
                """if not caller_flag:
                    used_edges.pop(n)"""
                    
        virtual_used_edges = {}
        for n in visitor.virtual_uses_edges:
            user_node = visited_nodes[n] if n in visited_nodes else None
            if user_node and user_node.defined and user_node.namespace is not None:
                caller_file = user_node.filename
                caller_type = user_node.flavor.value
                caller_flag = False
                virtual_used_edges[user_node] = {}
                for n2 in visitor.virtual_uses_edges[n]:
                    virtual_used_edges[user_node][n2] = visitor.virtual_uses_edges[n][n2]
                            
        import_used_edges = {}
        for n in visitor.import_uses_edges:
            user_node = visited_nodes[n] if n in visited_nodes else None
            if user_node and user_node.defined and user_node.namespace is not None:
                import_used_edges[user_node] = []
                for n2 in visitor.import_uses_edges[n]:
                    alias = n2
                    import_node = visitor.import_uses_edges[n][alias]
                    import_used_edges[user_node].append(import_node)
        
        qika = 1
        root_call_analyzer = cls(visited_nodes, define_edges, used_edges, virtual_used_edges, import_used_edges)
        return root_call_analyzer
    
class FolderMaker(object):
    """
    根据call_analyzer.py分析出的结果，构建与object原文件夹相同的结构，
    每个原py文件对应位置放置一个同名的json文件，文件内容为该py文件中的function的调用信息
    """
    def __init__(self, root_path):
        self.root_path = root_path
    
    def get_object_root(self):
        # 判断object的根目录，根据某一个module的路径来判断
        i=0
        dict_s = {}
        for node_name in self.call_analyzer.nodes:
            sample_node = self.call_analyzer.nodes[node_name]
            if sample_node.flavor.value == "module" and '.'  in sample_node.name:
                rela_path = sample_node.name.replace(".", "/")
                break
        object_root = sample_node.filename.split(rela_path, 1)[0].rstrip("/")
        
        return object_root

    def node_to_info(self, node):
        info = {
            "path": node.filename[len(self.object_root)+1:],
            "name": node.namespace+'.'+node.name if node.namespace else node.name,
            "type": node.flavor.value,
            "defined": node.defined,
        }
        """
        if hasattr(node.ast_node, "lineno"):
            info["position"] = [(node.ast_node.lineno, node.ast_node.col_offset),(node.ast_node.end_lineno, node.ast_node.end_col_offset)]
        """
        return info
        
    def virtual_to_info(self, name, condidate_list):
        info = {
            "name": name,
            "condidate": []
        }
        for condidate in condidate_list:
            info["condidate"].append({
            "path": condidate.filename[len(self.object_root)+1:] if condidate.filename else "",
            "name": condidate.namespace+'.'+condidate.name if condidate.namespace else condidate.name,
            "type": condidate.flavor.value,
            "defined": condidate.defined,
        })
        
        return info
    
    def process(self, call_analyzer, call_graph, object_root=None):
        file_list = {}
        self.call_analyzer = call_analyzer
        if object_root is None:
            self.object_root = self.get_object_root()
        else:
            self.object_root = object_root
        for caller in self.call_analyzer.used_edges.keys():
            file_path = caller.filename
            rela_path = file_path[len(self.object_root)+1:]
            caller_info = {"name": caller.name, 
                           "type": caller.flavor.value,
                           "namespace": caller.namespace, 
                           "position": [(0,0),(-1,-1)],
                           "body_position": [-1,-1],
                           "annotation": '',
                           "annotation_position": [-1,-1],
                           "in_class": sorted([self.node_to_info(n) for n in self.call_analyzer.used_edges[caller]["in_class"]], \
                               key=lambda x: x["name"]),
                           "in_file": sorted([self.node_to_info(n) for n in self.call_analyzer.used_edges[caller]["in_file"]], \
                               key=lambda x: x["name"]),
                           "in_object": sorted([self.node_to_info(n) for n in self.call_analyzer.used_edges[caller]["in_object"]], \
                               key=lambda x: x["name"]),
                           "virtual": sorted([self.virtual_to_info(key, value) for (key, value) in self.call_analyzer.virtual_used_edges[caller].items()], \
                               key=lambda x: x["name"]) if caller in self.call_analyzer.virtual_used_edges.keys() else [],
                           "import": sorted([self.node_to_info(n) for n in self.call_analyzer.import_used_edges[caller]], \
                                 key=lambda x: x["name"]) if caller in self.call_analyzer.import_used_edges.keys() else []
                           }
            
            if hasattr(caller.ast_node, "lineno"):
                caller_info["position"] = [(caller.ast_node.lineno, caller.ast_node.col_offset),(caller.ast_node.end_lineno, caller.ast_node.end_col_offset)]
            
            
            full_caller = self.node_to_info(caller)['name']
            if full_caller in call_graph.functional_info:
                caller_info["body_position"] = [call_graph.functional_info[full_caller]["body_begin"],
                                            call_graph.functional_info[full_caller]["body_end"]]
            
                if call_graph.functional_info[full_caller]["annotation"] != '':
                    caller_info["annotation"]  = call_graph.functional_info[full_caller]["annotation"]
                    caller_info["annotation_position"] = [call_graph.functional_info[full_caller]["annotation_begin"],
                                                        call_graph.functional_info[full_caller]["annotation_end"]]
                
            if rela_path not in file_list:
                file_list[rela_path] = {}
            file_list[rela_path][self.node_to_info(caller)['name']] = caller_info
            if 0 not in [len(caller_info[type_name]) for type_name in ['in_class','in_file','in_object','virtual']]:
                qika = caller_info
            
        for file_name in file_list.keys():
            # rela_path = file_name.split('.')[0].replace('/', '') + ".json"
            rela_path = file_name.split('.')[0] + ".json"
            # if rela_path[0] == '/':
            #     rela_path = rela_path[1:]
            file_path = os.path.join(self.root_path, rela_path)
            file_list[file_name] = dict(sorted(file_list[file_name].items(), key=lambda x: (x[1]["position"][0][0], x[1]["position"][0][1])))
            os.makedirs(file_path.rsplit(os.sep, 1)[0], exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(file_list[file_name], f, indent=4)
        # To accelerate the computation of Recall@k, we do not save the following two files in temporary folder
        # all_info_path = os.path.join(self.root_path, "all_call_info.json")
        # with open(all_info_path, "w") as f:
        #     json.dump(file_list, f, indent=4)
        # call_analyzer_path = os.path.join(self.root_path, "analyzer_result.pkl")
        # out_put = open(call_analyzer_path, 'wb')
        # tree_str = pickle.dumps(call_graph)
        # out_put.write(tree_str)
        # out_put.close()

            