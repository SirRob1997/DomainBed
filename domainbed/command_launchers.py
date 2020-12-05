# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import os
import re

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """Doesn't run anything; instead, prints each command.
    Useful for testing."""
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def get_trailing_numbers(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None

def slurm_launcher(commands):
    """Launch commands on the slurm cluster"""
    curr_counts = [get_trailing_numbers(name) for name in os.listdir("slurm/scripts")]
    if curr_counts:
        new_experiment = max(curr_counts) + 1
    else:
        new_experiment = 0
    experiment_path = f"slurm/scripts/exp{new_experiment}"
    os.makedirs(experiment_path)

    for i, cmd in enumerate(commands):
        curr_file = experiment_path + "/" + "exp-" + str(new_experiment)  + SCRIPT_NAME + str(i) + ".sh"
        with open(curr_file, "w", encoding='utf-8') as f:
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --ntasks=1 \n")
            for key, value in SLURM_REGISTRY.items():
                line = "#SBATCH --" + key + "=" + str(value) + "\n"
                f.write(line)
            f.write("\n")
            f.write("cd ~/" + PROJECT_DIR + "\n\n")
            f.write(cmd)
        print("Launched: " + curr_file)
        subprocess.call("sbatch " + curr_file + " --exclude=slurm-bm-60", shell=True)

SLURM_REGISTRY = {
    'cpus-per-task': 1,
    'nodes': 1,                                         # Requests that cores are on one node
    'time': '0-03:00',                                  # Max time per task, here 3 hours
    'partition': 'gpu-2080ti-dev',
#    'partition': 'gpu-v100',
    'gres': "gpu:1",
    'output': 'slurm/stdout/schmidt_%j.out',     # STD OUT APPEND FILE
    'error': 'slurm/err/schmidt_%j.out',         # ERROR APPEND FILE
#    'mail-type': 'FAIL',
#    'mail-user': 'rob.schmidt@student.uni-tuebingen.de'
}

PROJECT_DIR = "DomainBed/"
SCRIPT_NAME = "command"

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'slurm': slurm_launcher
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
