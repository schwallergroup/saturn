from models.generator import Generator
from hallucinated_memory.sequence_mutator import SequenceMutator
from hallucinated_memory.genetic_mutator import GeneticMutator
from hallucinated_memory.dataclass import HallucinatedMemoryParameters


def initialize_hallucinator(prior: Generator, parameters: HallucinatedMemoryParameters):
    """Initializes and returns the Hallucinator"""
    if parameters.hallucination_method.lower() == "sequence":
        return SequenceMutator(prior=prior,
                               num_hallucinations=parameters.num_hallucinations,
                               num_selected=parameters.num_selected,
                               selection_criterion=parameters.selection_criterion.lower())
                               
    elif parameters.hallucination_method.lower() == "ga":
        return GeneticMutator(prior=prior,
                              num_hallucinations=parameters.num_hallucinations,
                              num_selected=parameters.num_selected,
                              selection_criterion=parameters.selection_criterion.lower())
    else:
        raise NotImplementedError(f"Hallucination method {parameters.hallucination_method} not implemented.")
