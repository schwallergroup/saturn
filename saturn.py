"""
Parent script that executes Sample Efficient Generative Molecular Design using Memory Manipulation (SATURN).
Takes as input a JSON configuration file that specifies all parameters for the generatve experiment.
Adapted from https://github.com/MolecularAI/Reinvent/input.py.
"""
import json
import argparse

from oracles.oracle import Oracle
from oracles.dataclass import OracleConfiguration

from goal_directed_generation.reinforcement_learning import ReinforcementLearningAgent
from goal_directed_generation.dataclass import ReinforcementLearningParameters, GoalDirectedGenerationConfiguration
from experience_replay.dataclass import ExperienceReplayParameters
from hallucinated_memory.dataclass import HallucinatedMemoryParameters
from beam_enumeration.dataclass import BeamEnumerationParameters
from diversity_filter.dataclass import DiversityFilterParameters


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
        # TODO: lightning trainer, track NLL, track validity and apply randomization during training
        pass
    elif running_mode == "goal_directed_generation":
        # 1. Construct the Oracle
        oracle = Oracle(OracleConfiguration(**config["oracle"]))
        # 2. Construct the Reinforcement Learning Agent
        reinforcement_learning_agent = ReinforcementLearningAgent(
            oracle=oracle,
            configuration=GoalDirectedGenerationConfiguration(
                config["goal_directed_generation"]["seed"],
                ReinforcementLearningParameters(**config["goal_directed_generation"]["reinforcement_learning"]),
                ExperienceReplayParameters(**config["goal_directed_generation"]["experience_replay"]),
                DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
                HallucinatedMemoryParameters(**config["goal_directed_generation"]["hallucinated_memory"]),
                BeamEnumerationParameters(**config["goal_directed_generation"]["beam_enumeration"]),
            )
        )
        pass
    else:
        raise ValueError(f"Running mode: {running_mode} is not implemented.")