# -------------------------------
# Beam Enumeration implementation
# -------------------------------
from typing import Tuple, List, Set
import logging
import torch
import numpy as np
import json

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from models.generator import Generator
from beam_enumeration.reward_tracker import RewardTracker

# Import model architectures
from models.rnn import RNN
from models.decoder import Decoder
from models.mamba import MambaLMHead


class BeamEnumeration:
    def __init__(self,
        k: int = 2,
        beam_steps: int = 18,
        substructure_type: str = "structure",
        substructure_min_size: int = 15,
        pool_size: int = 4,
        pool_saving_frequency: int = 500,
        patience: int = 5,
        token_sampling_method: str = "topk",
        filter_patience_limit: int = 100000
    ):
        assert substructure_type in ["structure", "scaffold"], "substructure_type must be either 'structure' or 'scaffold'"
        assert token_sampling_method in ["topk", "sample"], "token_sampling_method must be either 'topk' or 'sample'"

        self.k = k
        self.beam_steps = beam_steps
        self.substructure_type = substructure_type
        self.substructure_min_size = substructure_min_size
        self.pool_size = pool_size

        # Used to track frequency of Beam Enumeration substructure saving
        self.last_save_multiple = 0
        self.pool_saving_frequency = pool_saving_frequency

        # Denotes how tokens are sampled - either top k or sampling from the distribution
        self.token_sampling_method = token_sampling_method
        
        # Track reward trajectory
        self.reward_tracker = RewardTracker(patience=patience)

        # Keep track of *all* scaffolds/substructures
        self.entire_pool = {}
        # Keep track of the most probable scaffolds/substructures
        self.pool = {}
        # Enforce probable substructures to contain heavy atoms
        self.heavy_atoms = {'N', 'n', 'O', 'o', 'S', 's'}

        # Track how many SMILES are filtered (based on Beam Enumeration) across the entire run
        self.filter_history = []

        # In case extremely improbable substructures are extracted
        self.filter_patience_limit = filter_patience_limit
        self.filter_patience = 0

    @torch.no_grad()
    def exhaustive_beam_expansion(self, agent: Generator) -> List[str]:
        """
        This method performs beam expansion to enumerate the set of highest probability (on average) sub-sequences.
        Total number of sub-sequences is k^beam_steps.
        """
        device = next(agent.network.parameters()).device
        # Start with k number of "start" sequences
        start_token = torch.zeros(self.k, dtype=torch.long, device=device)
        start_token[:] = agent.vocabulary["^"]
        input_vector = start_token
        hidden_state = None

        enumerated_sequences = [agent.vocabulary["^"] * torch.ones([self.k, 1], dtype=torch.long, device=device)]

        # Enumerate beam_steps number of time-steps
        for time_step in range(1, self.beam_steps + 1, 1):
            if isinstance(agent.network, RNN):
                logits, hidden_state = agent.network(input_vector.unsqueeze(1), hidden_state)
                logits = logits.squeeze(1)
            elif isinstance(agent.network, Decoder):
                # TODO: Implement
                pass
            elif isinstance(agent.network, MambaLMHead):
                # TODO: Implement
                pass

            probabilities = logits.softmax(dim=1)

            # If taking top k tokens
            if self.token_sampling_method == "topk":
                _, top_indices = torch.topk(probabilities, self.k)
            # If sampling tokens from the distribution
            elif self.token_sampling_method == "sample":
                top_indices = torch.multinomial(probabilities, self.k)

            # At time_step = 1, directly take the top k most probable tokens (or sampled)
            if time_step == 1:
                # Below is hard-coded to index 0 because at time-step 1,
                # The top k probabilities are always the same since hidden state = None
                # If using token sampling, take the 1st (index 0) set of top indices (even though the different indices may be different tokens due to stochasticity)
                top_indices = top_indices[0]
                enumerated_sequences = [torch.cat([start_token, first_token.unsqueeze(0)]) for start_token, first_token
                                        in zip(enumerated_sequences[0], top_indices)]
                # The input to the next time-step is the current time-steps' most probable tokens
                input_vector = top_indices
            # Otherwise, each sub-sequence needs to be extended by its top k (or sampled) tokens
            else:
                # Initialize a temporary list that stores all the sub-sequences from this time-step
                temp = []
                # For each sub-sequence, extend it by its most probable k tokens (or sampled)
                for sub_sequence, top_tokens in zip(enumerated_sequences, top_indices):
                    for token in top_tokens:
                        temp.append(torch.cat([sub_sequence, token.unsqueeze(0)]))

                # At this point, len(temp_sequences) > len(enumerated_sequences) because of the beam expansion
                enumerated_sequences = temp

                # The input to the next time-step is the current time-steps' most probable tokens
                input_vector = top_indices.flatten()

                # Duplicate hidden states for LSTM cell passing
                hidden_state = (hidden_state[0].repeat_interleave(self.k, dim=1), hidden_state[1].repeat_interleave(self.k, dim=1))

        # At this point, enumerated_sequences contains the most probable and
        # exhaustively enumerated token sequences - decode these into SMILES
        smiles = [agent.tokenizer.untokenize(agent.vocabulary.decode(seq.cpu().numpy())) for seq in enumerated_sequences]

        return smiles

    def filter_batch(
        self, 
        seqs: torch.Tensor,
        smiles: np.ndarray[str]
        ) -> Tuple[torch.Tensor, np.ndarray[str]]:
        """
        This method takes a generated batch of SMILES and returns only those
        that contain the most probable substructures as stored in *pool*.
        """
        indices = []
        for idx, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                for substructure in self.pool:
                    if mol.HasSubstructMatch(substructure):
                        indices.append(idx)
                        break

        return seqs[indices], smiles[indices]

    def pool_update(self, agent: Generator):
        """
        This method executes Beam Enumeration and extracts and stores the most frequent substructures in self.pool.
        """
        logging.info("Excecuting Beam Enumeration")
        smiles_subsequences = self.exhaustive_beam_expansion(agent)
        pool = self.get_top_substructures(smiles_subsequences)
        self.pool = pool

    def get_top_substructures(self, smiles_subsequences: List[str]) -> List[Mol]:
        """
        This method extracts the most frequent substructures from the enumerated SMILES subsequences.
        """
        # Clear the pool - important not to accumulate substructures from previous Beam Enumeration executions
        self.pool = {}
        for seq in smiles_subsequences:
            # Check whether to extract substructure itself or substructure scaffold
            if self.substructure_type == "structure":
                structures = self.substructure_extractor(seq)
            elif self.substructure_type == "scaffold":
                structures = self.scaffold_extractor(seq)
            # Not every subsequence has valid structures
            if len(structures) > 0:
                for s in structures:
                    # enforce minimum substructure size
                    if len(s) >= self.substructure_min_size:
                        if s not in self.pool:
                            self.pool[s] = 1
                        else:
                            self.pool[s] += 1
                    else:
                        continue

        # Sort frequency of substructures
        sorted_pool = dict(sorted(self.pool.items(), key=lambda x: x[1], reverse=True))
        # Store all substructures with their corresponding frequency
        self.entire_pool = sorted_pool
        # Store the most frequent substructures
        sliced_pool = list(sorted_pool.keys())[:self.pool_size]

        return [Chem.MolFromSmiles(s) for s in sliced_pool]

    def substructure_extractor(self, subsequence: str) -> Set[str]:
        """
        Extracts substructures from a SMILES subsequence.
        """
        # Use a Set for tracking to avoid repeated counting of the same substructure
        substructures = set()
        for idx in range(len(subsequence) - 2):
            mol = Chem.MolFromSmiles(subsequence[:3 + idx])
            if mol is not None:
                try:
                    canonical_substructure = Chem.MolToSmiles(mol, canonical=True)
                except Exception:
                    # Invariant Violation has been observed
                    continue
                canonical_chars = set(canonical_substructure)
                if self.contains_heavy_atoms(canonical_chars):
                    substructures.add(canonical_substructure)

        return substructures

    def scaffold_extractor(self, subsequence: str) -> Set[str]:
        """
        Extracts Bemis-Murcko scaffolds from a SMILES subsequence.
        """
        # Use a Set for tracking to avoid repeated counting of the same scaffold
        scaffolds = set()
        for idx in range(len(subsequence) - 2):
            mol = Chem.MolFromSmiles(subsequence[:3 + idx])
            if mol is not None:
                try:
                    # Store the Bemis-Murcko scaffold because heavy atoms are important
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                except Exception:
                    # Scaffold extraction may raise RDKit valency error - skip these for now
                    continue
                try:
                    canonical_scaffold = Chem.MolToSmiles(scaffold, canonical=True)
                except Exception:
                    # Invariant Violation has been observed
                    continue
                canonical_chars = set(canonical_scaffold)
                if self.contains_heavy_atoms(canonical_chars):
                    scaffolds.add(canonical_scaffold)

        return scaffolds
    
    def contains_heavy_atoms(self, substructure_chars: Set[str]) -> bool:
        """
        Checks whether a substructure contains heavy atoms.
        Substructures are only considered if they contain heavy atoms.
        """
        return len(substructure_chars.intersection(self.heavy_atoms)) > 0
    
    def epoch_updates(
        self, 
        agent: Generator, 
        num_valid_smiles: int, 
        mean_reward: float, 
        oracle_calls: int
    ) -> None:
        """
        This method performs 4 updates on every epoch:
        1. Updates self-conditioning filter history (track number of SMILES kept after filtering on pooled substructures)
        2. Check whether to execute Beam Enumeration - if yes, do so
        3. Check whether to write-out pooled substructures
        4. Resets the filter patience counter
        """
        # Track self-conditioned filtering
        self.filter_history.append(num_valid_smiles)
        # Check whether to execute Beam Enumeration
        if self.reward_tracker.is_beam_epoch(mean_reward):
            self.pool_update(agent)
        # Check whether to write-out the pooled substructures
        if (oracle_calls > self.pool_saving_frequency) and (oracle_calls // self.pool_saving_frequency > self.last_save_multiple):
            self.write_out_pool(oracle_calls)
            self.last_save_multiple = oracle_calls // self.pool_saving_frequency
        
        self.filter_patience = 0

    def filtered_epoch_updates(self):
        """
        This method performs 2 updates and executes when *none* of the sampled batch contains the Beam substructures (they are all discarded):
        1. Increments Filter Patience
        2. Appends 0 to Filter History
        """
        self.filter_patience += 1
        self.filter_history.append(0)

    def patience_limit_reached(self) -> bool:
        """
        Checks whether the filter patience limit has been reached.
        (consecutive generation epoch with all molecules discarded due to not containing the pooled substructure).
        """
        return self.filter_patience == self.filter_patience_limit
    
    def end_actions(self, oracle_calls: int) -> None:
        logging.info(f"Executed Beam Enumeration {self.reward_tracker.beam_executions} times")
        logging.info("Saving final pooled substructures")
        self.write_out_pool(oracle_calls)
        # Also write out entire pool
        self.write_out_entire_pool()
        # Write out Beam Enumeration self-conditioning history
        self.write_out_filtering()

    def write_out_pool(self, oracle_calls: int) -> None:
        write_pool = [Chem.MolToSmiles(mol) for mol in self.pool]
        with open(f"substructures_{oracle_calls}.smi", "w+") as f:
            for s in write_pool:
                f.write(f'{s}\n')

    def write_out_entire_pool(self) -> None:
        with open(f"entire_pool.json", "w") as f:
            json.dump(self.entire_pool, f, indent=2)

    def write_out_filtering(self) -> None:
        with open(f"filter_history.txt", "w") as f:
            for num in self.filter_history:
                f.write(f'{num}\n')
