"""
Adapted from https://github.com/MolecularAI/Reinvent with code additions for:
    1. Augmented Memory: https://pubs.acs.org/doi/10.1021/jacsau.4c00066
    2. Beam Enumeration: https://openreview.net/forum?id=7UhxsmbdaQ
    3. Hallucinated Memory (GraphGA based: https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)
"""
import os
import logging
import time
import torch
import numpy as np

import utils.chemistry_utils as chemistry_utils
from goal_directed_generation.utils import sample_unique_sequences
from utils.utils import to_tensor, setup_logging

from oracles.oracle import Oracle
from goal_directed_generation.dataclass import GoalDirectedGenerationConfiguration

from models.generator import Generator
from experience_replay.replay_buffer import ReplayBuffer
from diversity_filter.diversity_filter import DiversityFilter
from hallucinated_memory.utils import initialize_hallucinator
from beam_enumeration.beam_enumeration import BeamEnumeration


class ReinforcementLearningAgent:
    """
    RL agent for goal-directed generation.
    """
    def __init__(
        self, 
        logging_path: str,
        model_checkpoints_dir: str,
        oracle: Oracle,
        configuration: GoalDirectedGenerationConfiguration,
        device: str
    ):
        self.prior = Generator.load_from_file(configuration.reinforcement_learning.prior, device)
        # Prior model is not updated so disable gradients
        self._disable_prior_gradients()
        self.agent = Generator.load_from_file(configuration.reinforcement_learning.agent, device)
        self.device = self.agent.device
        # In case the Agent is to be trained on CPU, move also the Prior to CPU to avoid tensors on different devices
        self.prior.network.to(self.device)

        # Seed for documentation
        self.seed = configuration.seed

        # Oracle
        self.oracle = oracle

        # RL parameters
        self.batch_size = configuration.reinforcement_learning.batch_size
        self.learning_rate = configuration.reinforcement_learning.learning_rate
        self.sigma = configuration.reinforcement_learning.sigma
        self.augmented_memory = configuration.reinforcement_learning.augmented_memory
        self.augmentation_rounds = configuration.reinforcement_learning.augmentation_rounds
        self.selective_memory_purge = configuration.reinforcement_learning.selective_memory_purge

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(parameters=configuration.experience_replay)
        
        # Seed the Replay Buffer (if applicable)
        self.oracle = self.replay_buffer.prepopulate_buffer(self.oracle)

        # Diversity Filter
        self.diversity_filter = DiversityFilter(configuration.diversity_filter)

        # Hallucinated Memory
        self.execute_hallucinated_memory = configuration.hallucinated_memory.execute_hallucinated_memory
        self.hallucinator = initialize_hallucinator(
            prior=self.prior,
            parameters=configuration.hallucinated_memory
        )

        # Beam Enumeration
        self.execute_beam_enumeration = configuration.beam_enumeration.execute_beam_enumeration
        self.beam_enumeration = BeamEnumeration(
            k=configuration.beam_enumeration.beam_k,
            beam_steps=configuration.beam_enumeration.beam_steps,
            substructure_type=configuration.beam_enumeration.substructure_type.lower(),
            substructure_min_size=configuration.beam_enumeration.structure_min_size,
            pool_size=configuration.beam_enumeration.pool_size,
            pool_saving_frequency=configuration.beam_enumeration.pool_saving_frequency,
            patience=configuration.beam_enumeration.patience,
            token_sampling_method=configuration.beam_enumeration.token_sampling_method,
            filter_patience_limit=configuration.beam_enumeration.filter_patience_limit
        )

        # Only the Agent is updated so the Prior does not need an optimizer
        self.optimizer = torch.optim.AdamW(self.agent.get_network_parameters(), lr=self.learning_rate)
        # Model checkpointing save directory
        self.model_checkpoints_dir = model_checkpoints_dir
        os.makedirs(self.model_checkpoints_dir, exist_ok=True)
        self.logging_path = logging_path
        # Set up logging
        setup_logging(logging_path)
  
    def run(self):
        start_time = time.perf_counter()
        logging.info(f"Starting RL generative experiment with oracle budget: {self.oracle.budget}")
        # FIXME: could be dangerous in case of infinite loop
        while not self.oracle.budget_exceeded():
            logging.info(f"Oracle calls: {self.oracle.calls}/{self.oracle.budget}")

            # 1. Sample unique SMILES from the Agent
            seqs, smiles, _ = sample_unique_sequences(self.agent, self.batch_size)

            # 2. Beam Enumeration: Filter SMILES using the Beam Enumeration pool
            if (self.execute_beam_enumeration) and (len(self.beam_enumeration.pool) != 0):
                seqs, smiles = self.beam_enumeration.filter_batch(seqs, smiles)

            # 3. Beam Enumeration: If all SMILES are filtered, proceed to generate a new batch
            if len(smiles) == 0:
                self.beam_enumeration.filtered_epoch_updates()
                if self.beam_enumeration.patience_limit_reached():
                    logging.info("Beam Enumeration: Patience limit reached. Ending run (not indicative of experiment failing).")
                    break
                continue

            # 4. Oracle call on sampled batch
            #    Rewards are already penalized by the Diversity Filter
            smiles, penalized_rewards = self.oracle(smiles, self.diversity_filter)

            # 5. Beam Enumeration: Check whether to execute Beam Enumeration
            #    NOTE: Beam Enumeration execution criterion is based only on the *sampled* batch
            if self.execute_beam_enumeration:
                self.beam_enumeration.epoch_updates(
                    agent=self.agent,
                    num_valid_smiles=len(smiles),
                    mean_reward=penalized_rewards.mean(),
                    oracle_calls=self.oracle.calls
                )

            # 6. Hallucinated Memory: Hallucinate new SMILES from the Replay Buffer
            if (self.execute_hallucinated_memory) and (len(self.replay_buffer.memory) == self.replay_buffer.memory_size):
                hallucinated_smiles = self.hallucinator.hallucinate(self.replay_buffer.memory)
                # 7. Hallucinated Memory: Oracle call on hallucinated batch
                hallucinated_smiles, hallucinated_penalized_rewards = self.oracle(hallucinated_smiles, self.diversity_filter, is_hallucinated_batch=True)
                # 8. Update the hallucination history
                self.hallucinator.epoch_updates(
                    oracle_calls=self.oracle.calls,
                    buffer_rewards=self.replay_buffer.memory["reward"],
                    hallucinations=hallucinated_smiles,
                    hallucination_rewards=hallucinated_penalized_rewards
                )
            else:
                hallucinated_smiles, hallucinated_penalized_rewards = np.array([]), np.array([])

            # 9. Concatenate sampled batch with hallucinated batch
            # TODO: Hallucinated SMILES' loss could be scaled via Importance Sampling
            smiles = np.concatenate((smiles, hallucinated_smiles), 0)
            penalized_rewards = np.concatenate((penalized_rewards, hallucinated_penalized_rewards), 0)

            # 10. Compute the loss
            #     "smiles" contains the concatenated sampled and hallucinated SMILES
            loss = self.compute_loss(smiles, penalized_rewards)

            # 11. Update Replay Buffer
            self.replay_buffer.add(
                smiles=smiles, 
                rewards=penalized_rewards
            )

            # 12. Add experience replay to the loss
            #     NOTE: this is done *after* updating the Replay Buffer so new best-so-far sampled *and* hallucinated SMILES can be sampled
            er_smiles, er_rewards = self.replay_buffer.sample_memory()

            # 13. Compute the loss for the experience replay SMILES
            er_loss = self.compute_loss(er_smiles, er_rewards)
        
            # 14. Concatenate losses to get the total loss and backpropagate
            loss = torch.cat((loss, er_loss), 0)
            self.backpropagate(loss)

            # 15. Augmented Memory
            if self.augmented_memory and len(self.replay_buffer.memory) > 0:
                # NOTE: *Highly* recommended that Selective Memory Purge is enabled
                #       All penalized scaffolds are removed from the Replay Buffer *before* executing Augmented Memory
                if self.selective_memory_purge:
                    self.replay_buffer.selective_memory_purge(smiles, penalized_rewards)
                for _ in range(self.augmentation_rounds):
                    # Get randomized SMILES for both the sampled and hallucinated SMILES
                    randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, self.prior)
                    # Compute the loss
                    loss = self.compute_loss(randomized_smiles, penalized_rewards)
                    # Augmented Memory: Key operation for sample efficiency
                    randomized_buffer_smiles, randomized_buffer_rewards = self.replay_buffer.augmented_memory_replay(self.prior)
                    augmented_memory_loss = self.compute_loss(randomized_buffer_smiles, randomized_buffer_rewards)
                    loss = torch.cat((loss, augmented_memory_loss), 0)
                    self.backpropagate(loss)

        logging.info(f"Budget reached - final oracle calls: {self.oracle.calls}/{self.oracle.budget}")
        self.write_out_results()
        end_time = time.perf_counter()
        logging.info(f"Total wall time: {end_time - start_time} seconds.")
        self.agent.save(os.path.join(self.model_checkpoints_dir, f"final_{self.agent.model_architecture}_agent.ckpt"))

    def compute_loss(
        self, 
        smiles: np.ndarray[str],
        rewards: np.ndarray[float]
    ) -> torch.Tensor:
        """
        Compute the loss for the RL agent.
        Based on REINVENT's original loss function: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x
        """
        if len(smiles) != 0:
            prior_likelihoods = -self.prior.likelihood_smiles(smiles)
            agent_likelihoods = -self.agent.likelihood_smiles(smiles)
            augmented_likelihoods = prior_likelihoods + self.sigma * to_tensor(rewards, self.device)
            loss = torch.pow((augmented_likelihoods - agent_likelihoods), 2)
            return loss
        else:
            return torch.tensor([], dtype=torch.float64, device=self.device)

    def backpropagate(self, loss: torch.Tensor) -> None:
        """
        Agent update via backpropagation.
        Directly returns if the loss is empty.
        """
        if len(loss) > 0:
            loss = loss.mean()
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
        else:
            return

    def _disable_prior_gradients(self):
        """Disable gradients for the Prior as it is not updated."""
        for param in self.prior.get_network_parameters():
            param.requires_grad = False

    def write_out_results(self):
        """
        Writes out the following results:
            1. Oracle History
            2. Beam Enumeration History
            3. Hallucination History
            4. Number of Oracle repeats
        """
        base_save_path = os.path.dirname(self.logging_path)
        self.oracle.write_out_oracle_history(base_save_path)
        self.oracle.write_out_repeat_history(base_save_path)

        if self.execute_beam_enumeration:
            self.beam_enumeration.end_actions(self.oracle.calls)

        if self.execute_hallucinated_memory:
            self.hallucinator.write_out_history(base_save_path)
