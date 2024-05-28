from dataclasses import dataclass

@dataclass
class HallucinatedMemoryParameters:
    execute_hallucinated_memory: bool
    hallucination_method: str = "ga"
    num_hallucinations: int = 100
    num_selected: int = 10
    selection_criterion: str = "tanimoto_distance"
