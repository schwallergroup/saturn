from dataclasses import dataclass, field
from typing import Dict, List
from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters

@dataclass
class DockStreamParameters:
    name: str
    reward_shaping_function_parameters: RewardShapingFunctionParameters
    weight: float = 1.0
    configuration_path: str
    docker_script_path: str
    environment_path: str
