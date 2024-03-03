"""
Token-level mutation of SMILES strings.
Similar to the STONED algorithm: https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc00231g
"""
from typing import List
from hallucinated_memory.hallucinator import Hallucinator
import pandas as pd
import numpy as np
from rdkit import Chem
from copy import deepcopy
from utils.chemistry_utils import canonicalize_smiles


class SequenceMutator(Hallucinator):
    def __init__(
        self,
        prior,
        num_hallucinations: int = 100,
        num_selected: int = 10,
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

        # The set of actions that a hallucination can be generated from
        self.action_set = [
            "mutate", 
            "insert", 
            "delete"
        ]
        # How many times an action can be taken 
        self.action_count = [1, 2, 3]

        # Store the hallucination history
        self.hallucination_history = pd.DataFrame({})
        
    def hallucinate(self, buffer: pd.DataFrame) -> np.ndarray[str]:
        """
        Hallucinate a set of unique SMILES strings similar to the STONED algorithm.
        """
        # Denote the parent the highest reward molecule in the buffer
        parent = list(buffer["smiles"].iloc[0])

        # Hallucinate a set of unique SMILES
        hallucinated_set = set()

        while len(hallucinated_set) != self.num_hallucinations:
            print(len(hallucinated_set))
            # Choose mutation action
            action = np.random.choice(self.action_set)
            # Choose number of times to execute the action
            num_actions = np.random.choice(self.action_count)

            # Hallucinate
            hallucinated_smiles = self.get_hallucinated_smiles(parent, action, num_actions)
            # Add canonicalized SMILES to guarantee uniqueness
            hallucinated_set.add(canonicalize_smiles(hallucinated_smiles))

        return self.select_hallucinations(
            parent=parent,
            hallucinations=[Chem.MolFromSmiles(s) for s in hallucinated_set],
            hallucinations_smiles=hallucinated_set
        )

    def get_hallucinated_smiles(
        self, 
        parent: List[str],
        action: str, 
        num_actions: int
    ) -> str:
        """
        Returns a hallucinated SMILES based on the parent SMILES and the mutation action.
        """
        tries = 0
        while tries < 1000:
            if action == "mutate":
                hallucinated_smiles = self.mutate(deepcopy(parent), num_actions)
            elif action == "insert":
                hallucinated_smiles = self.insert(deepcopy(parent), num_actions)
            elif action == "delete":
                hallucinated_smiles = self.delete(deepcopy(parent), num_actions)

            hallucinated_mol = Chem.MolFromSmiles(hallucinated_smiles)
            if self.can_be_encoded(hallucinated_mol, self.tokenizer, self.vocabulary):
                return hallucinated_smiles

            tries += 1

        # If all 1000 tries produced invalid molecules, return the parent
        return "".join(parent)

    def mutate(
        self, 
        parent: List[str], 
        num_actions: int
    ) -> str:
        """
        Swap a random token in the parent SMILES string with a random token from the Vocabulary.
        """
        # Keep track of which position is mutated so all mutation locations are unique
        mutated_indices = set()
        while len(mutated_indices) != num_actions:
            mutation_index = np.random.choice(list(range(len(parent))))
            if mutation_index not in mutated_indices:
                parent[mutation_index] = np.random.choice(self.tokens)
                mutated_indices.add(mutation_index)

        return "".join(parent)

    def insert(
        self, 
        parent: List[str],
        num_actions: int
    ) -> str:
        """
        Insert a random token from the Vocabulary into a random position in the parent SMILES string.
        """
        for _ in range(num_actions):
            insert_index = np.random.choice(list(range(len(parent))))
            parent.insert(insert_index, np.random.choice(self.tokens))

        return "".join(parent)

    @staticmethod
    def delete(
        parent: List[str],
        num_actions: int
    ) -> str:
        """
        Delete a random token from the parent SMILES string.
        """
        for _ in range(num_actions):
            delete_index = np.random.choice(list(range(len(parent))))
            del parent[delete_index]

        return "".join(parent)
