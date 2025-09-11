import os, json
import textwrap
from tree_sitter import Language, Parser

def create_parser(language):
    parser = Parser()
    parser.set_language(Language('build/my-languages.so', language))
    return parser

def predict_relationship(candidate_name: str, attributes: dict):
    if attributes['type'] == 'function':
        file_path = '.'.join(attributes['namespace'].split('.')[:-1])
        if candidate_name.startswith(file_path):
            return 'intra_file'
        else:
            return 'cross_file'
    elif attributes['type'] == 'method':
        class_path = '.'.join(attributes['namespace'].split('.')[:-1])
        if candidate_name.startswith(class_path):
            return 'intra_class'
        else:
            file_path = '.'.join(attributes['namespace'].split('.')[:-2])
            # file_name = attributes['namespace'].split('.')[0]
            if candidate_name.startswith(file_path):
                return 'intra_file'
            else:
                return 'cross_file'

def extract_dependency(attributes: dict):
    dependency = {'intra_class': [], 'intra_file': [], 'cross_file': []}
    for dep in attributes['in_class']:
        dependency['intra_class'].append(dep['name'])
    for dep in attributes['in_file']:
        dependency['intra_file'].append(dep['name'])
    for dep in attributes['in_object']:
        dependency['cross_file'].append(dep['name'])
    return dependency

def find_json_file(path: str):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.json'):
                file_list.append(os.path.join(root, file))
    return file_list

def load_json_data(input_file: str):
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            data.append(js)
    return data

def extract_code_by_position(code_path, signature_position, body_position):
    with open(code_path, 'r') as f:
        lines = f.readlines()
    code_lines = lines[signature_position[0]-1:body_position[1]]
    return ''.join(code_lines)

def retrieve_requirement_file(project_path: str):
    requirement_files = []
    if os.path.exists(os.path.join(project_path, 'requirements.txt')):
        requirement_files.append('requirements.txt')
    if os.path.exists(os.path.join(project_path, 'requirements')):
        requirement_dir = os.path.join(project_path, 'requirements')
        for file in os.listdir(requirement_dir):
            if file.endswith('.txt'):
                requirement_files.append(os.path.join('requirements', file))
    return requirement_files

def count_indent(args, data):
    code_file_path = os.path.join(args.source_code_root, data['completion_path'])
    with open(code_file_path, 'r') as f:
        lines = f.readlines()
    body_first_line = lines[data['body_position'][0]-1]
    indent = len(body_first_line) - len(textwrap.dedent(body_first_line))
    return indent

def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

def traverse_ast(root_node):
    node_list = []
    def traverse(node):
        node_list.append(node)
        for child in node.children:
            traverse(child)
    traverse(root_node)
    return node_list


def search_import(node):
    """
    Search all import or from_import nodes from a root node.
    """
    import_nodes = []
    for child in node.children:
        if child.type == 'import_statement' or child.type == 'import_from_statement':
            import_nodes.append(child)
        result = search_import(child)
        if result:
            import_nodes.extend(result)
    return import_nodes