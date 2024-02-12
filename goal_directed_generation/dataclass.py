from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ReinforcementLearningParameters:
    reinforcement_learning: Dict[str, List[str]] = field(default_factory=dict)
    

@dataclass
class GoalDirectedGenerationParameters:
    reinforcement_learning: Dict[str, List[str]] = field(default_factory=dict)
    experience_replay: Dict[str, List[str]] = field(default_factory=dict)
    hallucinated_memory: Dict[str, List[str]] = field(default_factory=dict)
    beam_enumeration: Dict[str, List[str]] = field(default_factory=dict)
    diversity_filter: 
