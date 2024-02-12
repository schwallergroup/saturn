from dataclasses import dataclass

@dataclass
class DiversityFilterParameters:
    name: str = "IdenticalMurckoScaffold"
    bucket_size: int = 10
    min_similarity: float = 0.4
