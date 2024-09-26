from dataclasses import dataclass

@dataclass
class EnforcedBlocksDiversityFilterParameters:
    # Similar to the Diversity Filter Based on REINVENT 3.2: https://github.com/MolecularAI/Reinvent
    # TODO: The dataclass is currently empty - future granular control can add a specific bucket size for each enforced block
    enforced_building_blocks_file: str
    bucket_size: int = 300
