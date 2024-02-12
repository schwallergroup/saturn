from typing import List
from dataclasses import dataclass
from oracles.reward_shaping.function_parameters import RewardShapingFunctionParameters

@dataclass
class OracleComponentParameters:
    name: str
    weight: float
    # preliminary_check is a flag to indicate whether an OracleComponent should be executed first.
    # If this OracleComponent is not satisfied, the SMILES is discarded. 
    # The intended use case is for MPO objectives with expensive components, e.g., MD, 
    # where the other objectives are cheap to compute and should be satisfied, e.g., MW < 500 Da.
    preliminary_check: bool = False
    # specific parameters contain oracle-specific parameters, e.g., reference SMILES for Tanimoto similarity
    specific_parameters: dict = {}
    reward_shaping_function_parameters: RewardShapingFunctionParameters

@dataclass
class OracleConfiguration:
    aggregator: str = "product"
    components: List[OracleComponentParameters]
