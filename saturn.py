"""
Parent script that executes Sample Efficient Generative Molecular Design using Memory Manipulation (SATURN).
Takes as input a JSON configuration file that specifies all parameters for the generatve experiment.
Adapted from https://github.com/MolecularAI/Reinvent/input.py.
"""
import json
import argparse

from oracles.oracle import Oracle

parser = argparse.ArgumentParser(description="Run SATURN.")
parser.add_argument(
    "config", 
    type=str,
    help="Path to the JSON configuration file."
)

def read_json_file(path: str):
    with open(path) as f:
        json_input = f.read().replace("\r", "").replace("\n", "")
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")

if __name__ == "__main__":
    args = parser.parse_args()
    
    config = read_json_file(args.config)
    running_mode = config["running_mode"].lower()

    if running_mode == "distribution_learning":
        # TODO: execute distribution learning (either pre-training or fine-tuning)
        pass
    elif running_mode == "goal_directed_generation":
        # 1. Construct the Oracle
        oracle = Oracle(config["oracle"])
        exit()
        # 2. Construct the Reinforcement Learning Agent

        pass
    else:
        raise ValueError(f"Running mode: {config.running_mode} is not implemented.")