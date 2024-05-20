from dataclasses import dataclass

@dataclass
class DiversityFilterParameters:
    # Based on REINVENT 3.2: https://github.com/MolecularAI/Reinvent
    # TODO: "name" is not currently used. All scaffolds are automatically defined as Bemis-Murcko scaffolds
    name: str = "IdenticalMurckoScaffold"
    bucket_size: int = 10
