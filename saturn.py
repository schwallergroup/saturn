"""
Parent script that executes Sample Efficient Generative Molecular Design using Memory Manipulation (SATURN).
Takes as input a JSON configuration file that specifies all parameters for the generatve experiment.
Adapted from https://github.com/MolecularAI/Reinvent/input.py.
"""
import time
import json
import argparse
#import torch
#from utils.utils import set_seed_everywhere

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
    start_time = time.perf_counter()
    args = parser.parse_args()
    
    config = read_json_file(args.config)
    running_mode = config["running_mode"].lower()

    if running_mode == "distribution_learning":
        # TODO: execute distribution learning (either pre-training or fine-tuning)
        # TODO: lightning trainer, track NLL, track SMILES validity, and don't forget to apply randomization during training (or have the option to)
        pass
    elif running_mode == "goal_directed_generation":
        # 1. (Optionally) set the seed
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = config["goal_directed_generation"]["seed"]
        #set_seed_everywhere(seed, device)


        # 2. Construct the Oracle
        oracle = Oracle(OracleConfiguration(**config["oracle"]))

        # 3. Construct the Reinforcement Learning Agent
        reinforcement_learning_agent = ReinforcementLearningAgent(
            oracle=oracle,
            configuration=GoalDirectedGenerationConfiguration(
                seed,
                ReinforcementLearningParameters(**config["goal_directed_generation"]["reinforcement_learning"]),
                ExperienceReplayParameters(**config["goal_directed_generation"]["experience_replay"]),
                DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
                HallucinatedMemoryParameters(**config["goal_directed_generation"]["hallucinated_memory"]),
                BeamEnumerationParameters(**config["goal_directed_generation"]["beam_enumeration"]),
            )
        )

        # 4. Run Goal-Directed Generation via Reinforcement Learning
        reinforcement_learning_agent.run()
        end_time = time.perf_counter()
        print(f"Wall time: {end_time - start_time:.2f}s")
    else:
        raise ValueError(f"Running mode: {running_mode} is not implemented.")