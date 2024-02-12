from dataclasses import dataclass

@dataclass
class RewardShapingFunctionParameters:
    # defaults to no transformation
    transformation_function: str = "no_transformation"
    parameters: dict = {}
