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

for task_id in range(len(data)):
    if os.path.exists(os.path.join(save_dir, f"results_{task_id}.jsonl")):
        continue
    project = data[task_id]["project"]

    os.system(f"sudo docker run --rm \
    -v {pred_dir}:/pred_dir:ro \
    -v {save_dir}:/rs_dir \
    -v {repo_dir}:/input:ro \
    -v {repo_dir}/data_with_test_case:/output:ro \
    -v {repo_dir}/{project}/:/package:ro \
    codeeval-runner --task_id {task_id} --problem_file /pred_dir/processed_generations.json --rs_dir /rs_dir --timeout 120")