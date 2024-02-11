from dataclasses import dataclass

@dataclass
class RewardShapingFunctionParameters:
    # default to no transformation
    transformation_function: str = "no_transformation"
    parameters: dict = {}
