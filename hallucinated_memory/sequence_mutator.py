"""
Token-level mutation of SMILES strings.
"""

from hallucinated_memory.hallucinator import Hallucinator
import pandas as pd
import numpy as np
from rdkit import Chem
from copy import deepcopy


class SequenceMutator(Hallucinator):
    def __init__(self,
                 prior,
                 num_hallucinations=100,
                 # TODO: num_hallucinations != num_selected because of potential duplicates
                 #       and also because of the selection criterion (in case it's not random)
                 num_selected=10,
                 selection_criterion: str="random"):
        
        self.vocabulary = prior.vocabulary
        self.tokenizer = prior.tokenizer
        self.tokens = self.vocabulary.tokens()

        # how many hallucinations to generate
        self.num_hallucinations = num_hallucinations
        # how many hallucinations to return
        self.num_selected = num_selected
        # how to select the hallucinations to return
        self.selection_criterion = selection_criterion

        # the set of actions that a hallucination can be generated from
        self.action_set = [
            "mutate", 
            "insert", 
            "delete"
        ]
        # how many times an action can be taken 
        self.action_count = [1, 2, 3]

        # store the hallucination history
        self.hallucination_history = pd.DataFrame({})
        
    def hallucinate(self, buffer: pd.DataFrame) -> np.ndarray[str]:
        # denote the parent the highest reward molecule in the buffer
        parent = list(buffer["smiles"].iloc[0])

        # hallucinate a set of unique SMILES
        # TODO: canonicalize?
        hallucinated_set = set()

        while len(hallucinated_set) != self.num_hallucinations:
            print(len(hallucinated_set))
            # choose mutation action
            action = np.random.choice(self.action_set)
            # choose number of times to execute the action
            num_actions = np.random.choice(self.action_count)

            # hallucinate
            hallucinated_smiles = self.get_hallucinated_smiles(parent, action, num_actions)
            hallucinated_set.add(hallucinated_smiles)

        if self.selection_criterion == "random":
            # FIXME: at the moment, 100 hallucinations are generated, but only 10 are returned
            return np.random.choice(list(hallucinated_set), self.num_selected)
        else:
            raise NotImplementedError("Only random selection is currently implemented.")

    def get_hallucinated_smiles(self, parent: list, action: str, num_actions: int) -> str:
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

        # if all 1000 tries produced invalid molecules, return the parent
        return "".join(parent)

    def mutate(self, parent: list, num_actions: int) -> str:
        # keep track of which position is mutated so all mutation locations are unique
        mutated_indices = set()
        while len(mutated_indices) != num_actions:
            mutation_index = np.random.choice(list(range(len(parent))))
            if mutation_index not in mutated_indices:
                parent[mutation_index] = np.random.choice(self.tokens)
                mutated_indices.add(mutation_index)

        return "".join(parent)

    def insert(self, parent: list, num_actions: int) -> str:
        for _ in range(num_actions):
            insert_index = np.random.choice(list(range(len(parent))))
            parent.insert(insert_index, np.random.choice(self.tokens))

        return "".join(parent)

    @staticmethod
    def delete(parent: list, num_actions: int) -> str:
        for _ in range(num_actions):
            delete_index = np.random.choice(list(range(len(parent))))
            del parent[delete_index]

        return "".join(parent)
