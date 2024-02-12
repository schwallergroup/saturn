# -------------------------------
# Beam Enumeration implementation
# -------------------------------

import torch
import pandas as pd
import numpy as np
import json

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from running_modes.beam_enumeration.reward_tracker import RewardTracker
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase

class BeamEnumeration:
    def __init__(self,
                 k: int,
                 beam_steps: int,
                 substructure_type: str,
                 substructure_min_size: int,
                 pool_size: int,
                 pool_saving_frequency: int,
                 patience: int,
                 token_sampling_method: str,
                 filter_patience_limit: int):
        self.k = k
        self.beam_steps = beam_steps
        self.substructure_type = substructure_type
        self.substructure_min_size = substructure_min_size
        self.pool_size = pool_size
        # used to track frequency of Beam Enumeration substructure saving
        self.last_save_multiple = 0
        self.pool_saving_frequency = pool_saving_frequency
        # denotes how tokens are sampled - either top k or sampling from the distribution
        self.token_sampling_method = token_sampling_method
        
        # track reward trajectory
        self.reward_tracker = RewardTracker(patience=patience)
        # keep track of *all* scaffolds/substructures
        self.entire_pool = {}
        # keep track of the most probable scaffolds/substructures
        self.pool = {}
        # enforce probable substructures to contain heavy atoms
        self.heavy_atoms = {'N', 'n', 'O', 'o', 'S', 's'}

        # track how many SMILES are filtered (based on Beam Enumeration) across the entire run
        self.filter_history = []

        # in case extremely improbable substructures are extracted
        self.filter_patience_limit = filter_patience_limit
        self.filter_patience = 0

    @torch.no_grad()
    def exhaustive_beam_expansion(self, agent):
        """this method performs beam expansion to enumerate the set of highest probability (on average) sub-sequences. Total number of sub-sequences is k^beam_steps."""

        # start with k number of "start" sequences
        start_token = torch.zeros(self.k, dtype=torch.long)
        start_token[:] = agent.vocabulary["^"]
        input_vector = start_token
        hidden_state = None

        enumerated_sequences = [agent.vocabulary["^"] * torch.ones([self.k, 1], dtype=torch.long)]

        # enumerate beam_steps number of time-steps
        for time_step in range(1, self.beam_steps + 1, 1):
            logits, hidden_state = agent.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)

            # if taking top k tokens
            if self.token_sampling_method == 'topk':
                _, top_indices = torch.topk(probabilities, self.k)
            # if sampling tokens from the distribution
            elif self.token_sampling_method == 'sample':
                top_indices = torch.multinomial(probabilities, self.k)

            # at time_step = 1, directly take the top k most probable tokens (or sampled)
            if time_step == 1:
                # below is hard-coded to index 0 because at time-step 1,
                # the top k probabilities are always the same since hidden state = None
                # if using token sampling, take the 1st (index 0) set of top indices (even though the different indices may be different tokens due to stochasticity)
                top_indices = top_indices[0]
                enumerated_sequences = [torch.cat([start_token, first_token.unsqueeze(0)]) for start_token, first_token
                                        in zip(enumerated_sequences[0], top_indices)]
                # the input to the next time-step is the current time-steps' most probable tokens
                input_vector = top_indices
            # otherwise, each sub-sequence needs to be extended by its top k (or sampled) tokens
            else:
                # initialize a temporary list that stores all the sub-sequences from this time-step
                temp = []
                # for each sub-sequence, extend it by its most probable k tokens (or sampled)
                for sub_sequence, top_tokens in zip(enumerated_sequences, top_indices):
                    for token in top_tokens:
                        temp.append(torch.cat([sub_sequence, token.unsqueeze(0)]))

                # at this point, len(temp_sequences) > len(enumerated_sequences) because of the beam expansion
                enumerated_sequences = temp

                # the input to the next time-step is the current time-steps' most probable tokens
                input_vector = top_indices.flatten()

                # duplicate hidden states for LSTM cell passing
                hidden_state = (hidden_state[0].repeat_interleave(self.k, dim=1), hidden_state[1].repeat_interleave(self.k, dim=1))

        # at this point, enumerated_sequences contains the most probable and
        # exhaustively enumerated token sequences - decode these into SMILES
        smiles = [agent.tokenizer.untokenize(agent.vocabulary.decode(seq.cpu().numpy())) for seq in enumerated_sequences]

        return smiles

    def filter_batch(self, seqs, smiles, agent_likelihood):
        """
        this method takes a generated batch of SMILES and returns only those
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

        return seqs[indices], smiles[indices], agent_likelihood[indices]

    def pool_update(self, agent):
        """this method performs Beam Enumeration and extracts and stores the most frequent substructures in *self.pool*."""
        print('----- Performing Beam Enumeration -----')
        subsequences = self.exhaustive_beam_expansion(agent)
        pool = self.get_top_substructures(subsequences)

        # store the RDKit Mol objects
        self.pool = [Chem.MolFromSmiles(s) for s in pool]

    def get_top_substructures(self, subsequences: list) -> list:
        # clear pool
        self.pool = {}
        for seq in subsequences:
            # check whether to extract substructure itself or substructure scaffold
            if self.substructure_type == 'structure':
                structures = self.substructure_extractor(seq)
            else:
                structures = self.scaffold_extractor(seq)
            # not every subsequence has valid structures
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

        # sort frequency of substructures
        sorted_pool = dict(sorted(self.pool.items(), key=lambda x: x[1], reverse=True))
        # store all substructures with their corresponding frequency
        self.entire_pool = sorted_pool
        # get the most frequent substructures
        sliced_pool = list(sorted_pool.keys())[:self.pool_size]

        return sliced_pool

    def substructure_extractor(self, subsequence: str) -> set:
        # use a Set for tracking to avoid repeated counting of the same substructure
        substructures = set()
        for idx in range(len(subsequence) - 2):
            mol = Chem.MolFromSmiles(subsequence[:3 + idx])
            if mol is not None:
                canonical_substructure = Chem.MolToSmiles(mol, canonical=True)
                canonical_chars = set(canonical_substructure)
                if len(canonical_chars.intersection(self.heavy_atoms)) > 0:
                    substructures.add(canonical_substructure)

        return substructures

    def scaffold_extractor(self, subsequence: str) -> set:
        # use a Set for tracking to avoid repeated counting of the same scaffold
        scaffolds = set()
        for idx in range(len(subsequence) - 2):
            mol = Chem.MolFromSmiles(subsequence[:3 + idx])
            if mol is not None:
                try:
                    # store the Bemis-Murcko scaffold because heavy atoms are important
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                except Exception:
                    # scaffold extraction may raise RDKit valency error - skip these for now
                    continue
                # get canonical
                canonical_scaffold = Chem.MolToSmiles(scaffold, canonical=True)
                canonical_chars = set(canonical_scaffold)
                if len(canonical_chars.intersection(self.heavy_atoms)) > 0:
                    scaffolds.add(canonical_scaffold)

        return scaffolds
    
    def epoch_updates(self, agent: GenerativeModelBase, num_valid_smiles: int, reward: float, oracle_calls: int):
        """
        this method performs 4 updates on every epoch:
        1. Updates self-conditioning filter history (track number of SMILES kept after filtering on pooled substructures)
        2. Checks whether to execute Beam Enumeration - if yes, do so
        3. Check whether to write-out pooled substructures
        4. Resets the filter patience counter
        """
        # track self-conditioned filtering
        self.filter_history.append(num_valid_smiles)
        # check whether to execute Beam Enumeration
        if self.reward_tracker.is_beam_epoch(reward):
            self.pool_update(agent)
        # check whether to write-out the pooled substructures
        if (oracle_calls > self.pool_saving_frequency) and (oracle_calls // self.pool_saving_frequency > self.last_save_multiple):
            self.write_out_pool(oracle_calls)
            self.last_save_multiple = oracle_calls // self.pool_saving_frequency
        
        self.filter_patience = 0
    
    def end_actions(self, oracle_calls):
        print(f'Executed Beam Enumeration {self.reward_tracker.beam_executions} times')
        print('Saving final pooled substructures')
        self.write_out_pool(oracle_calls)
        # also write out entire pool
        self.write_out_entire_pool()
        # write out Beam Enumeration self-conditioning history
        self.write_out_filtering()

    def filtered_epoch_updates(self):
        """
        this method performs 2 updates and executes when *none* of the sampled batch contains the Beam substructures (they are all discarded):
        1. Increments filter_patience
        2. Appends 0 to filter_history
        """
        self.filter_patience += 1
        self.filter_history.append(0)

    def patience_limit_reached(self):
        return self.filter_patience == self.filter_patience_limit

    def write_out_pool(self, oracle_calls):
        write_pool = [Chem.MolToSmiles(mol) for mol in self.pool]
        with open(f'substructures_{oracle_calls}.smi', 'w+') as f:
            for s in write_pool:
                f.write(f'{s}\n')

    def write_out_entire_pool(self):
        with open(f'entire_pool.json', 'w+') as f:
            json.dump(self.entire_pool, f, indent=2)

    def write_out_filtering(self):
        with open(f'filter_history.txt', 'w+') as f:
            for num in self.filter_history:
                f.write(f'{num}\n')
