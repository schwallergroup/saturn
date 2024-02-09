from rdkit import Chem
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from running_modes.hallucination.sequence_mutator import SequenceMutator
from running_modes.hallucination.genetic_mutator import GeneticMutator
from running_modes.configurations import HallucinatedMemoryConfiguration


def initialize_hallucinator(prior: GenerativeModelBase, hallucination_config: HallucinatedMemoryConfiguration):
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
