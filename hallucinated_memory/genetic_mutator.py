"""
Apply a genetic algorithm to a SMILES string to generate new SMILES strings.
Uses GraphGA's algorithm: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c
"""
from typing import List
import logging
from hallucinated_memory.hallucinator import Hallucinator
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from hallucinated_memory.graphga_utils import crossover, mutate

from utils.chemistry_utils import canonicalize_smiles


class GeneticMutator(Hallucinator):
    def __init__(
        self,
        prior,
        num_hallucinations: int = 100,
        num_selected: int = 5,
        selection_criterion: str = "random"
    ):
        self.vocabulary = prior.vocabulary
        self.tokenizer = prior.tokenizer
        self.tokens = self.vocabulary.get_tokens()

        # How many hallucinations to generate
        self.num_hallucinations = num_hallucinations
        # How many hallucinations to return
        self.num_selected = num_selected
        # How to select the hallucinations to return
        self.selection_criterion = selection_criterion

        # Store the hallucination history
        self.hallucination_history = pd.DataFrame({})

    def hallucinate(self, buffer: pd.DataFrame) -> np.ndarray[str]:
        """
        Hallucinate a set of unique SMILES strings via GraphGA's algorithm.
        """
        hallucinated_set = set()

        # Consider the buffer the population
        parents = [Chem.MolFromSmiles(s) for s in buffer["smiles"]]
        parents_rewards = buffer["reward"].values

        # To avoid infinite loop, set a maximum number of iterations
        tries = 0

        while (len(hallucinated_set) != self.num_hallucinations) and (tries < 10000):
            # Generate child
            # TODO: Can add a mutation rate class attribute
            child = reproduce(parents, parents_rewards, 0.1)
            if self.can_be_encoded(child, self.tokenizer, self.vocabulary):
                # Add canonicalized SMILES to guarantee uniqueness
                hallucinated_set.add(canonicalize_smiles(Chem.MolToSmiles(child)))
            tries += 1

        if len(hallucinated_set) != self.num_hallucinations:
            logging.info(f"Hallucinated Memory: Generated {len(hallucinated_set)}/{self.num_hallucinations} valid hallucinations in 10000 attempts")

        return self.select_hallucinations(
            parent=parents,
            hallucinations=[Chem.MolFromSmiles(s) for s in hallucinated_set],
            hallucinations_smiles=hallucinated_set
        )
    
@staticmethod
def choose_parents(
    parents: List[Mol], 
    parents_rewards: np.ndarray[float]
) -> np.ndarray[Mol]:
    """
    Sample 2 parents with probability proportional to their rewards.
    Based on implementation from GEAM: https://anonymous.4open.science/r/GEAM-45EF/utils_ga/ga.py
    """
    sampling_probs = [reward / sum(parents_rewards) for reward in parents_rewards]
    parents = np.random.choice(parents, p=sampling_probs, size=2)
    return parents

@staticmethod
def reproduce(
    parents: List[Mol], 
    parents_rewards: np.ndarray[float], 
    mutation_rate: float
) -> Mol:
    """
    Select 2 parents, crossover, and mutate to generate a child.
    """
    parent_1, parent_2 = choose_parents(parents, parents_rewards)
    child = crossover.crossover(parent_1, parent_2)
    if child is not None:
        child = mutate.mutate(child, mutation_rate)

    return child
