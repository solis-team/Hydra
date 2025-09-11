import os, datasets
import json
import shlex
import subprocess
import logging

import argparse
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--subset', default='full_context', help='full_context | medium_context | short_context')
# parser.add_argument('--run_gt', help='Execute the grouth truth solution', action="store_true")

parser.add_argument('--gendir', help='Raw Gen folder', type=str, default=None)


args = parser.parse_args()
pred_dir = os.path.join(args.gendir, "predictions")
pred_dir = os.path.abspath(pred_dir)
save_dir = os.path.join(args.gendir, "execution_rs")
save_dir = os.path.abspath(save_dir)




if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)


data = datasets.load_dataset("Fsoft-AIC/RepoExec")
data = data[args.subset]
repo_dir = os.path.abspath(os.path.dirname(os.getcwd()))
print("Repo root dir:", repo_dir)
print("All tasks:", len(data))

task_project_map = {i: data[i]["project"] for i in range(len(data))}

project_processed_count = {}
for filename in os.listdir(save_dir):
    if filename.startswith("results_") and filename.endswith(".jsonl"):
        task_id = int(filename.replace("results_", "").replace(".jsonl", ""))
        if task_id in task_project_map:
            project = task_project_map[task_id]
            project_processed_count[project] = project_processed_count.get(project, 0) + 1


project_containers = {}  
project_sample_count = {} 

for task in data:
    project = task["project"]
    if project not in project_sample_count:
        project_sample_count[project] = 0
    project_sample_count[project] += 1

for task_id in range(len(data)):
    if os.path.exists(os.path.join(save_dir, f"results_{task_id}.jsonl")):
        continue
    
    project = data[task_id]["project"]
    project_path = os.path.join(repo_dir, project)
    
    if project not in project_processed_count:
        project_processed_count[project] = 0
    
    is_tornado = "tornado" in project.lower()
    if is_tornado:
        run_cmd = f"""docker run --rm \
            -v {shlex.quote(pred_dir)}:/pred_dir:ro \
            -v {shlex.quote(save_dir)}:/rs_dir \
            -v {shlex.quote(repo_dir)}:/input:ro \
            -v {shlex.quote(repo_dir)}/data_with_test_case:/output:ro \
            -v {shlex.quote(project_path)}:/package:ro \
            tornado-runner \
            --task_id {task_id} \
            --problem_file /pred_dir/processed_generations.json \
            --rs_dir /rs_dir \
            --timeout 120"""
    else:
        safe_project_name = project.replace('/', '-').replace('.', '-')
        container_name = f"codeeval-{safe_project_name}"
        project_containers[project] = container_name
        check_container_cmd = f"docker ps -a --filter name={container_name} --format '{{{{.ID}}}}'"
        container_id = os.popen(check_container_cmd).read().strip()
        if container_id:  
            check_running_cmd = f"docker ps --filter id={container_id} --format '{{{{.ID}}}}'"
            running_id = os.popen(check_running_cmd).read().strip()
            if not running_id:  
                logger.info(f"container {container_name} exists but is not running. Restarting")
                start_cmd = f"docker start {container_id}"
                os.system(start_cmd)
                time.sleep(2)
            else:
                logger.info(f"using existing running container:{container_name}")
        else:  
            create_cmd = f"""docker run -d --name {container_name} \
                -v {shlex.quote(pred_dir)}:/pred_dir:ro \
                -v {shlex.quote(save_dir)}:/rs_dir \
                -v {shlex.quote(repo_dir)}:/input:ro \
                -v {shlex.quote(repo_dir)}/data_with_test_case:/output:ro \
                -v {shlex.quote(project_path)}:/package:ro \
                -v {container_name}_env:/usr/local/lib/python3.10/site-packages \
                --entrypoint /bin/bash \
                codeeval-runner -c "if [ ! -f /tmp/.env_setup_done ]; then pip install -r /package/requirements.txt && touch /tmp/.env_setup_done; fi; tail -f /dev/null" """
            logger.info(f"Creating container with persistent volume for project {project}")
            result = os.system(create_cmd)
            if result != 0:
                logger.error(f"Error when creating container for {project}: {result}")
                continue
        run_cmd = f"""docker run --rm \
            --volumes-from {container_name} \
            codeeval-runner \
            --task_id {task_id} \
            --problem_file /pred_dir/processed_generations.json \
            --rs_dir /rs_dir \
            --timeout 120"""
    logger.info(f"Running task {task_id} for project {project}")
    os.system(run_cmd)
    project_processed_count[project] = project_processed_count.get(project) + 1
    if (not is_tornado) and project in project_containers:
        check_container_cmd = f"docker ps -a --filter name={project_containers[project]} --format '{{{{.ID}}}}'"
        container_id = os.popen(check_container_cmd).read().strip()
        if container_id and project in project_sample_count and project_processed_count[project] >= project_sample_count[project]:
            cleanup_cmd = f"docker stop {container_id} && docker rm {container_id}"
            os.system(cleanup_cmd)
            if project in project_containers:
                del project_containers[project]

