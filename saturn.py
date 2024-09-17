"""
Parent script that executes Sample Efficient Generative Molecular Design using Memory Manipulation (Saturn).
Takes as input a JSON configuration file that specifies all parameters for the generatve experiment.
Adapted from https://github.com/MolecularAI/Reinvent/input.py.
"""
import json
import argparse
from utils.utils import set_seed_everywhere

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

# Oracle (for Goal-Directed Generation)
from oracles.oracle import Oracle
from oracles.dataclass import OracleConfiguration

# Scoring
from scoring.scorer import Scorer
from scoring.dataclass import ScoringConfiguration

parser = argparse.ArgumentParser(description="Run Saturn.")
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

    # Set the seed
    device = config["device"]
    seed = config["seed"]
    set_seed_everywhere(seed, device)

    model_architecture = config["model_architecture"]

    if running_mode == "distribution_learning":
        # 1. Construct the Distribution Learning Trainer
        distribution_learning_trainer = DistributionLearningTrainer(
            logging_path=config["logging"]["logging_path"],
            model_checkpoints_dir=config["logging"]["model_checkpoints_dir"],
            configuration=DistributionLearningConfiguration(
                seed,
                model_architecture,
                **config["distribution_learning"]["parameters"])
        )
        # 2. Run Distribution Learning
        distribution_learning_trainer.run()

    elif running_mode == "goal_directed_generation":
        # 1. Construct the Oracle
        oracle = Oracle(OracleConfiguration(**config["oracle"]))

        # 2. Construct the Reinforcement Learning Agent
        reinforcement_learning_agent = ReinforcementLearningAgent(
            logging_frequency=config["logging"]["logging_frequency"],
            logging_path=config["logging"]["logging_path"],
            model_checkpoints_dir=config["logging"]["model_checkpoints_dir"],
            oracle=oracle,
            configuration=GoalDirectedGenerationConfiguration(
                seed,
                model_architecture,
                ReinforcementLearningParameters(**config["goal_directed_generation"]["reinforcement_learning"]),
                ExperienceReplayParameters(**config["goal_directed_generation"]["experience_replay"]),
                DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
                HallucinatedMemoryParameters(**config["goal_directed_generation"]["hallucinated_memory"]),
                BeamEnumerationParameters(**config["goal_directed_generation"]["beam_enumeration"]),
            ),
            device=device
        )

        # 3. Run Goal-Directed Generation via Reinforcement Learning
        reinforcement_learning_agent.run()

    elif running_mode in ["scoring", "scorer"]:
        # 1. Construct the Oracle
        oracle = Oracle(OracleConfiguration(**config["oracle"]))

        # 2. Construct the Scorer
        scorer = Scorer(
            config["logging"]["logging_path"],
            oracle=oracle,
            # FIXME: Currently required because the Oracle takes as input a Diversity Filter - remove this dependency
            diversity_filter_configuration=DiversityFilterParameters(**config["goal_directed_generation"]["diversity_filter"]),
            configuration=ScoringConfiguration(**config["scoring"])
        )

        # 3. Run Scoring
        scorer.run()

    else:
        raise ValueError(f"Running mode: {running_mode} is not implemented.")
