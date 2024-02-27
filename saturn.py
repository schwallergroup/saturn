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

# Distribution Learning
from distribution_learning.distribution_learning import DistributionLearningTrainer
from distribution_learning.dataclass import DistributionLearningConfiguration

# Goal-Directed Generation
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
    # TODO: self.logger to log messages instead of printing --> have a logging dir and output everything there
    start_time = time.perf_counter()
    args = parser.parse_args()
    
    config = read_json_file(args.config)
    running_mode = config["running_mode"].lower()

    # FIXME: train with seed too
    # (Optionally) set the seed
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config["seed"]
    model_type = config["model_type"]
    #set_seed_everywhere(seed, device)

    # TODO: logging should have results path that is *shared* for distribution learning and goal-directed generation
    if running_mode == "distribution_learning":
        distribution_learning_trainer = DistributionLearningTrainer(
            DistributionLearningConfiguration(
                seed,
                model_type,
                **config["distribution_learning"]["parameters"])
        )
        distribution_learning_trainer.run()
        # TODO: execute distribution learning (either pre-training or fine-tuning)
        # TODO: lightning trainer, track NLL, track SMILES validity, and don't forget to apply randomization during training (or have the option to)
        pass
    elif running_mode == "goal_directed_generation":
        # 1. Construct the Oracle
        oracle = Oracle(OracleConfiguration(**config["oracle"]))

        # 2. Construct the Reinforcement Learning Agent
        reinforcement_learning_agent = ReinforcementLearningAgent(
            oracle=oracle,
            configuration=GoalDirectedGenerationConfiguration(
                seed,
                model_type,
                ReinforcementLearningParameters(**config["goal_directed_generation"]["reinforcement_learning"]),
                ExperienceReplayParameters(**config["goal_directed_generation"]["experience_replay"]),
                DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
                HallucinatedMemoryParameters(**config["goal_directed_generation"]["hallucinated_memory"]),
                BeamEnumerationParameters(**config["goal_directed_generation"]["beam_enumeration"]),
            )
        )

        # 3. Run Goal-Directed Generation via Reinforcement Learning
        reinforcement_learning_agent.run()
    else:
        raise ValueError(f"Running mode: {running_mode} is not implemented.")
    
    end_time = time.perf_counter()
    print(f"Wall time: {end_time - start_time:.2f}s")
