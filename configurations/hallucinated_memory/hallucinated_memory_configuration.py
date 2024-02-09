from dataclasses import dataclass

@dataclass
class HallucinatedMemoryConfiguration:
    execute_hallucinated_memory: bool = True
    hallucination_method: str = "mutator"
    num_hallucinations: int = 100
    num_selected: int = 10
    selection_criterion: str = "random",
