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
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyan_zyf_v2.analyzer import CallGraphVisitor
from pyan_zyf_v2.call_analyzer import CallAnalyzer, FolderMaker
import fire

# TODO: use an int argument for verbosity
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename='func_call.log',
                    filemode='w')

#logger.addHandler(logging.StreamHandler())

def init_arguments():
    usage = """%(prog)s FILENAME... [--dot|--tgf|--yed|--svg|--html]"""
    desc = (
        "Analyse one or more Python source files and generate an"
        "approximate call graph of the modules, classes and functions"
        " within them."
    )

    parser = ArgumentParser(usage=usage, description=desc)

    parser.add_argument("--dot", action="store_true", default=False, help="output in GraphViz dot format")

    parser.add_argument("--tgf", action="store_true", default=False, help="output in Trivial Graph Format")

    parser.add_argument("--svg", action="store_true", default=False, help="output in SVG Format")

    parser.add_argument("--html", action="store_true", default=False, help="output in HTML Format")

    parser.add_argument("--yed", action="store_true", default=False, help="output in yEd GraphML Format")

    parser.add_argument("--file", dest="filename", help="write graph to FILE", metavar="FILE", default=None)

    parser.add_argument("--namespace", dest="namespace", help="filter for NAMESPACE", metavar="NAMESPACE", default=None)

    parser.add_argument("--function", dest="function", help="filter for FUNCTION", metavar="FUNCTION", default=None)

    parser.add_argument("-l", "--log", dest="logname", help="write log to LOG", metavar="LOG")

    parser.add_argument("-v", "--verbose", action="store_true", default=False, dest="verbose", help="verbose output")

    parser.add_argument(
        "-V",
        "--very-verbose",
        action="store_true",
        default=False,
        dest="very_verbose",
        help="even more verbose output (mainly for debug)",
    )

    parser.add_argument(
        "-d",
        "--defines",
        action="store_true",
        dest="draw_defines",
        help="add edges for 'defines' relationships [default]",
    )

    parser.add_argument(
        "-n",
        "--no-defines",
        action="store_false",
        default=True,
        dest="draw_defines",
        help="do not add edges for 'defines' relationships",
    )

    parser.add_argument(
        "-u",
        "--uses",
        action="store_true",
        default=True,
        dest="draw_uses",
        help="add edges for 'uses' relationships [default]",
    )

    parser.add_argument(
        "-N",
        "--no-uses",
        action="store_false",
        default=True,
        dest="draw_uses",
        help="do not add edges for 'uses' relationships",
    )

    parser.add_argument(
        "-c",
        "--colored",
        action="store_true",
        default=False,
        dest="colored",
        help="color nodes according to namespace [dot only]",
    )

    parser.add_argument(
        "-G",
        "--grouped-alt",
        action="store_true",
        default=False,
        dest="grouped_alt",
        help="suggest grouping by adding invisible defines edges [only useful with --no-defines]",
    )

    parser.add_argument(
        "-g",
        "--grouped",
        action="store_true",
        default=False,
        dest="grouped",
        help="group nodes (create subgraphs) according to namespace [dot only]",
    )

    parser.add_argument(
        "-e",
        "--nested-groups",
        action="store_true",
        default=False,
        dest="nested_groups",
        help="create nested groups (subgraphs) for nested namespaces (implies -g) [dot only]",
    )

    parser.add_argument(
        "--dot-rankdir",
        default="TB",
        dest="rankdir",
        help=(
            "specifies the dot graph 'rankdir' property for "
            "controlling the direction of the graph. "
            "Allowed values: ['TB', 'LR', 'BT', 'RL']. "
            "[dot only]"
        ),
    )

    parser.add_argument(
        "--dot-ranksep",
        default="0.5",
        dest="ranksep",
        help=(
            "specifies the dot graph 'ranksep' property for "
            "controlling desired rank separation, in inches. "
            "Allowed values: [0.02 .. 1000.0]. "
            "[dot only]"
        ),
    )

    parser.add_argument(
        "--graphviz-layout",
        default="dot",
        dest="layout",
        help=(
            "specifies the graphviz 'layout' property for "
            "the name of the layout algorithm to use. "
            "Allowed values: ['dot', 'neato', 'fdp', 'sfdp', 'twopi', 'circo']. "
            "Recommended values: ['dot', 'fdp']. "
            "[graphviz only]"
        ),
    )

    parser.add_argument(
        "-a",
        "--annotated",
        action="store_true",
        default=False,
        dest="annotated",
        help="annotate with module and source line number",
    )

    parser.add_argument(
        "--root",
        default=None,
        dest="root",
        help="Package root directory. Is inferred by default.",
    )
    
    return parser

def find_py_files(folder):
    py_files = []
    for root, dirs, files in os.walk(folder):
        if True in [item.startswith('.') or item == 'myenv' for item in root.split(os.sep)]:
            continue
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files

def process(target_object, Dataset_root, Dataset_Call_root):
    rela_path = os.path.relpath(target_object,Dataset_root)
    target_root = os.path.join(Dataset_Call_root, rela_path)
    target_files = find_py_files(target_object)

    filenames = []
    for fn in target_files:
        for fn2 in glob(fn, recursive=True):
            abs_fn2 = os.path.abspath(fn2)
            filenames.append(abs_fn2)
    

    v = CallGraphVisitor(filenames, logger=logger, root=None)

    graph = CallAnalyzer.from_visitor(v, target_root, logger=logger)
    folder_maker = FolderMaker(target_root)
    folder_maker.process(graph, v, target_object)

def main(project_root, output_root, cli_args=None):
    
    parser = init_arguments()
    known_args, unknown_args = parser.parse_known_args(cli_args)

    if known_args.nested_groups:
        known_args.grouped = True

    if known_args.logname:
        handler = logging.FileHandler(known_args.logname)
        logger.addHandler(handler)

    logger.debug(f"[files] {unknown_args}")

    if not os.path.exists(project_root):
        raise FileNotFoundError(f"{project_root} not found")
    
    finished_projects = []
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    else:
        for topic in os.listdir(output_root):
            topic_path = os.path.join(output_root, topic)
            for project in os.listdir(topic_path):
                finished_projects.append(project)
    print("Number of finished projects: "+str(len(finished_projects)))

    # print(os.listdir(project_root))
    TODO_projects = []
    for topic in os.listdir(project_root):
        topic_path = os.path.join(project_root, topic)
        for project in os.listdir(topic_path):
            if project not in finished_projects:
                project_path = os.path.join(topic_path, project)
                TODO_projects.append(project_path)
    print("Number of TODO projects: "+str(len(TODO_projects)))
    # print(TODO_projects[:5])
    # raise Exception("stop")
    

    done_num = 0
    with tqdm(total=len(TODO_projects)) as pbar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_proj = [
                executor.submit(process, target_object, project_root, output_root)
                for target_object in TODO_projects
            ]
            for future in as_completed(future_to_proj):
                pbar.update()
                done_num += 1
    print("Finish "+str(done_num)+" projects")
        

if __name__ == "__main__":
    fire.Fire(main)
