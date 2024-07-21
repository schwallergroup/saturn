from dataclasses import dataclass

@dataclass
class ScoringConfiguration:
    smiles_path: str
    output_csv_path: str
