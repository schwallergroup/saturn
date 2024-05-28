from abc import ABC, abstractmethod
from typing import Union, List, Set
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import TanimotoSimilarity, BulkTanimotoSimilarity


class Hallucinator(ABC):
    """Base class for hallucination strategies."""

    @abstractmethod
    def hallucinate(self, buffer: pd.DataFrame) -> np.array:
        raise NotImplementedError("Hallucinator must implement hallucinate method.")
    
    def epoch_updates(
        self, 
        oracle_calls: int,
        buffer_rewards: pd.Series,
        hallucinations: np.ndarray[str], 
        hallucination_rewards: np.ndarray[float]
    ) -> None:
        """
        Track the hallucination history of the parent sequence:
            1. Oracle calls so far
            2. Highest buffer reward (before hallucinating)
            3. Hallucinated SMILES
            4. Hallucinated rewards
            5. Whether a hallucination has a higher reward than the top buffer SMILES
            6. Number of hallucinations that were added to the buffer
        """
        highest_buffer_reward = buffer_rewards.max()
        df = pd.DataFrame(
            {
                "oracle_calls": [oracle_calls],
                "top_buffer_reward_before_hallucinating": [highest_buffer_reward],
                "hallucinations": [hallucinations],
                "hallucination_rewards": [hallucination_rewards],
                "hallucination_better_than_top_buffer": [np.any(hallucination_rewards > highest_buffer_reward)],
                "num_hallucinations_added_to_buffer": self.num_buffer_replacements(hallucination_rewards, buffer_rewards)
            }
        )
        self.hallucination_history = pd.concat([self.hallucination_history, df], ignore_index=True)
        
    def select_hallucinations(
        self,
        parent: Union[Mol, List[Mol]], 
        hallucinations: List[Mol],
        hallucinations_smiles: Set[str]
    ) -> np.ndarray[str]:
        """
        Selects a sub-set of hallucinations to return based on the selection criterion.
        1. Random: random selection
        2. Tanimoto Distance: select the most *dissimilar* hallucinations to the parent sequence by Tanimoto Similarity.
           Hypothesis: Mitigates mode collapse and increases Augmentation tolerability threshold
                       * But are the rewards worse?
        """
        if self.selection_criterion == "random":
            return np.random.choice(list(hallucinations_smiles), self.num_selected)
        elif self.selection_criterion == "tanimoto_distance":
            tanimoto_distances = []

            # If Sequence-based Hallucinator, Tanimoto distance relative to the parent sequence
            if isinstance(parent, Chem.Mol):
                parent_fp = GetMorganFingerprintAsBitVect(parent, radius=3, nBits=2048)
                for h in hallucinations:
                    h_fp = GetMorganFingerprintAsBitVect(h, radius=3, nBits=2048)
                    tanimoto_distances.append(TanimotoSimilarity(parent_fp, h_fp))
        
            # If GA-based Hallucinator, Tanimoto distance relative to entire buffer
            elif isinstance(parent, list):
                parent_fps = [GetMorganFingerprintAsBitVect(p, radius=3, nBits=2048) for p in parent]
                for h in hallucinations:
                    h_fp = GetMorganFingerprintAsBitVect(h, radius=3, nBits=2048)
                    # average Tanimoto distance of each hallucination to the entire buffer
                    tanimoto_distances.append(np.mean(BulkTanimotoSimilarity(h_fp, parent_fps)))
            
            # Return the most *dissimilar* molecules
            dissimilar_indices = np.argsort(tanimoto_distances)[:self.num_selected]
            return np.array(list(hallucinations_smiles))[dissimilar_indices]
            
        else:
            raise NotImplementedError(f"Selection criterion: {self.selection_criterion} not implemented.")

    def num_buffer_replacements(
        self, 
        hallucination_rewards: np.ndarray[float],
        buffer_rewards: pd.Series
    ) -> int:
        """
        Returns the number of times a batch of hallucinations have corresponding rewards better than the SMILES in the Replay Buffer.
        """
        # TODO: Optimize the implementation - brute force for now
        count = 0
        buffer_rewards = list(buffer_rewards)

        for reward in hallucination_rewards:
            for idx, buffer_reward in enumerate(buffer_rewards):
                if reward > buffer_reward:
                    count += 1
                    buffer_rewards.insert(idx, reward)
                    buffer_rewards.pop()
                    break

        return count
    
    def can_be_encoded(
        self,
        mol: Mol, 
        tokenizer, 
        vocabulary
    ) -> bool:
        """
        Checks whether a hallucinated molecule satisfies:
            1. It is a RDKit-parsable molecule
            2. It can be sanitized by RDKit
            3. It can be tokenized and encoded based on the model's vocabulary
        """
        if mol is not None:
            try:
                sanitized_mol = Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
                tokens = tokenizer.tokenize(Chem.MolToSmiles(mol, canonical=True))
                # ensure tokens can be encoded
                encoded = [vocabulary.encode(token) for token in tokens]
                return True
            except Exception:
                return False
        else:
            return False
    
    def write_out_history(self, path: str):
        self.hallucination_history.to_csv(os.path.join(path, "hallucination_history.csv"))
