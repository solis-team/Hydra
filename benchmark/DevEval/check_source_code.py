import os, sys
from subprocess import run

source_code_root = sys.argv[1]

topic_list = os.listdir(source_code_root)
for topic in topic_list:
    topic_path = os.path.join(source_code_root, topic)
    if not os.path.isdir(topic_path):
        continue
    project_list = os.listdir(topic_path)
    for project in project_list:
        project_path = os.path.join(topic_path, project)
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    if 'tmp_' + file in files:
                        print(os.path.join(root, file))
                        print(os.path.join(root, 'tmp_' + file))
                        print('---------------------')
                        run(['mv', os.path.join(root, 'tmp_' + file), os.path.join(root, file)])