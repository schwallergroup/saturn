# FIXME: change this
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from hallucinated_memory.sequence_mutator import SequenceMutator
from hallucinated_memory.genetic_mutator import GeneticMutator
from hallucinated_memory.dataclass import HallucinatedMemoryParameters


def initialize_hallucinator(prior: GenerativeModelBase, hallucination_config: HallucinatedMemoryParameters):
    """Initializes and returns the Hallucinator"""
    if hallucination_config.hallucination_method.lower() == "sequence":
        return SequenceMutator(prior=prior,
                               num_hallucinations=hallucination_config.num_hallucinations,
                               num_selected=hallucination_config.num_selected,
                               selection_criterion=hallucination_config.selection_criterion.lower())
                               
    elif hallucination_config.hallucination_method.lower() == "ga":
        return GeneticMutator(prior=prior,
                              num_hallucinations=hallucination_config.num_hallucinations,
                              num_selected=hallucination_config.num_selected,
                              selection_criterion=hallucination_config.selection_criterion.lower())
    else:
        raise NotImplementedError(f"Hallucination method {hallucination_config.hallucination_method} not implemented.")
