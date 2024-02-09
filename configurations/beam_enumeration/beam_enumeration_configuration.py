from dataclasses import dataclass

@dataclass
class BeamEnumerationConfiguration:
    execute_beam_enumeration: bool = True
    beam_k: int = 2
    beam_steps: int = 18
    substructure_type: str = "structure"
    structure_min_size: int = 15
    pool_size: int = 4
    pool_saving_frequency: int = 1000
    patience: int = 5
    token_sampling_method: str = "topk"
    filter_patience_limit: int = 50000
