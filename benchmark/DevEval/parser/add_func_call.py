"""
分析Object主文件夹，对其中每个class和func判别其调用了哪些外部资源（例如其他文件内的function）

使用pyan分析object文件夹，修正其中会引起bug的问题
    lambda节点
    
Text processing/xmnlp
"""

from argparse import ArgumentParser
from glob import glob
import logging
import os
import traceback
from tqdm import tqdm
import func_timeout
from func_timeout import func_set_timeout
import dill as pickle

from pyan_zyf_v2.analyzer import CallGraphVisitor
from pyan_zyf_v2.call_analyzer import CallAnalyzer, FolderMaker
from pyan_zyf_v2.visgraph import VisualGraph
from pyan_zyf_v2.writers import DotWriter, HTMLWriter, SVGWriter, TgfWriter, YedWriter
from pyan_zyf_v2.anutils import (
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


# TODO: use an int argument for verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='func_call.log',
                    filemode='w')

def find_py_files(folder):
    py_files = []
    for root, dirs, files in os.walk(folder):
        if True in [item.startswith('.') for item in root.split(os.sep)]:
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

folder_path = "path/to/folder"
py_files = find_py_files(folder_path)


def process(target_object,func_object_root, func_path, analyzer_result, target_root):

    with open(func_path, 'r') as f:
        func_content = f.read()
    
    with open(analyzer_result, 'rb') as analyzer:
        v: CallGraphVisitor = pickle.loads(analyzer.read())
    
    virtual_path = func_path.replace(func_object_root, target_object)
    
    v.add_process_one(virtual_path, content=func_content)
    v.postprocess()
    
    # 找到func_path对应的namespace
    namespace = get_module_name(virtual_path, root=None)

    graph = CallAnalyzer.from_visitor(v, target_root, prefix=namespace,logger=logger)
    folder_maker = FolderMaker(target_root)
    folder_maker.process(graph, v, target_object)

