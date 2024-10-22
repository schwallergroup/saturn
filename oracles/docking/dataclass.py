from dataclasses import dataclass
from typing import List
from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters

@dataclass
class DockStreamParameters:
    name: str
    reward_shaping_function_parameters: RewardShapingFunctionParameters
    weight: float
    configuration_path: str
    docker_script_path: str
    environment_path: str

@dataclass
class ConstrainedDockingParameters:
    hbind_binary: str
    enforce_interactions: bool
    reward_type: str
    enforced_residues: List[str]
    # TODO: Add other types of interactions to enforce
