import subprocess
import argparse


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--gendir', help='Raw Gen folder', type=str, default="data/generation/RepoExec")

scripts = [
    r"process-result.py",
    r"execute.py",
    r"passk.py",
    r"getdir.py"]

args= argument_parser.parse_args()
for script in scripts:
    print(f"running script: {script}")
    subprocess.run(["python", script, "--gendir", args.gendir])