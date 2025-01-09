from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EnforcedBuildingBlocksParameters:
    enforce_blocks: bool  # Whether to enforce building blocks
    enforced_building_blocks_file: str  # Path to the file containing the building blocks to enforce
    enforce_start: bool  # Whether to enforce building blocks at the starting-material node
    use_dense_reward: bool  # Whether to use dense reward
    reward_type: str  # Reward type
    tango_weights: Dict[str, float]  # TANGO weights

@dataclass
class EnforcedReactionsParameters:
    enforce_rxn_class_presence: bool  # Whether to enforce reaction class presence
    enforce_all_reactions: bool  # Whether to enforce all reactions in the route to be in the enforced rxn classes
    rxn_insight_env_name: str  # Name of the Rxn-INSIGHT environment
    enforced_rxn_classes: List[str]  # List of reaction classes to enforce
    avoid_rxn_classes: List[str]  # List of reaction classes to *avoid*
    rxn_info_extraction_script_path: str  # Path to the Rxn-INSIGHT extraction script
    seed_reactions: bool  # Whether to seed the route with enforced reaction classes
    seed_building_blocks_file: str  # Path to the file containing the building blocks to use for seeding
    seed_reactions_file_folder: str # Path to the folder where we store the preloaded reactions file for the seeding