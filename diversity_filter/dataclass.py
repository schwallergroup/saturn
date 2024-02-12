from dataclasses import dataclass

@dataclass
class DiversityFilterParameters:
    minsimilarity: float = 0.4
    name: str = "IdenticalMurckoScaffold"
    nbmax: int = 10
