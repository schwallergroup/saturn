from dataclasses import dataclass
from experience_replay.dataclass import ExperienceReplayParameters
from hallucinated_memory.dataclass import HallucinatedMemoryParameters
from beam_enumeration.dataclass import BeamEnumerationParameters
from diversity_filter.dataclass import DiversityFilterParameters

@dataclass
class ReinforcementLearningParameters:
    prior: str
    agent: str
    batch_size: int
    learning_rate: float = 0.0001
    margin_threshold: int
    sigma: int = 128
    augmented_memory: bool = True
    augmentation_rounds: int
    selective_memory_purge: bool = True

@dataclass
class GoalDirectedGenerationParameters:
    reinforcement_learning: ReinforcementLearningParameters
    experience_replay: ExperienceReplayParameters
    hallucinated_memory: HallucinatedMemoryParameters
    beam_enumeration: BeamEnumerationParameters
    diversity_filter: DiversityFilterParameters
