from dataclasses import dataclass

@dataclass
class ScoringConfiguration:
    smiles_path: str
    sample: bool
    agent: str
    sample_num: int
    output_csv_path: str
