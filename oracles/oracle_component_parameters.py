from dataclasses import dataclass
from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters

@dataclass
class OracleComponentParameters:
    name: str
    weight: float
    # specific parameters contain oracle-specific parameters
    specific_parameters: dict = {}
    reward_shaping_function_parameters: RewardShapingFunctionParameters
