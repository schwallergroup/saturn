"""
Apply a genetic algorithm to a SMILES string to generate new SMILES strings.
Uses Graph GA's algorithm: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c
"""

from hallucinated_memory.hallucinator import Hallucinator
import pandas as pd
import numpy as np
from rdkit import Chem
from hallucinated_memory.graphga_utils import crossover, mutate
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase



class GeneticMutator(Hallucinator):
    def __init__(
            self,
            prior: GenerativeModelBase,
            num_hallucinations: int=100,
            num_selected: int=10,
            selection_criterion: str="random"
            ):
        self.vocabulary = prior.vocabulary
        self.tokenizer = prior.tokenizer
        self.tokens = self.vocabulary.tokens()

        # how many hallucinations to generate
        self.num_hallucinations = num_hallucinations
        # how many hallucinations to return
        self.num_selected = num_selected
        # how to select the hallucinations to return
        self.selection_criterion = selection_criterion

        # store the hallucination history
        self.hallucination_history = pd.DataFrame({})

        # TODO: be able to fix seed for reproducibility?

    def hallucinate(self, buffer: pd.DataFrame) -> np.array:
        # hallucinate a set of unique SMILES
        # TODO: canonicalize?
        hallucinated_set = set()

        # consider the buffer the population
        parents = [Chem.MolFromSmiles(s) for s in buffer["smiles"]]
        parents_rewards = buffer["score"].values

        # to avoid infinite loop, set a maximum number of iterations
        tries = 0

        while (len(hallucinated_set) != self.num_hallucinations) and (tries < 1000):
            # generate child
            child = reproduce(parents, parents_rewards, mutation_rate=0.1)
            if self.can_be_encoded(child, self.tokenizer, self.vocabulary):
                hallucinated_set.add(Chem.MolToSmiles(child))
            tries += 1

        if len(hallucinated_set) != self.num_hallucinations:
            print(f"WARNING: generated {len(hallucinated_set)}/{self.num_hallucinations} valid hallucinations in 1000 attempts")

        return self.select_hallucinations(parent=parents,
                                          hallucinations=[Chem.MolFromSmiles(s) for s in hallucinated_set],
                                          hallucinations_smiles=hallucinated_set)


def choose_parents(parents, parents_rewards):
    # sample 2 parents with probability proportional to their rewards
    # based on GEAM: https://anonymous.4open.science/r/GEAM-45EF/utils_ga/ga.py
    # sum(sampling_probs) ≈ 1.0
    sampling_probs = [reward / sum(parents_rewards) for reward in parents_rewards]
    parents = np.random.choice(parents, p=sampling_probs, size=2)
    return parents


def reproduce(parents, parents_rewards, mutation_rate):
    parent_1, parent_2 = choose_parents(parents, parents_rewards)
    child = crossover.crossover(parent_1, parent_2)
    if child is not None:
        child = mutate.mutate(child, mutation_rate)
    return child