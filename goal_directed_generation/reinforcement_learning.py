"""
Adapted from https://github.com/MolecularAI/Reinvent with code additions for:
    1. Augmented Memory: https://chemrxiv.org/engage/chemrxiv/article-details/646a353da32ceeff2d014776
    2. Hallucinated Memory
    3. Beam Enumeration: https://arxiv.org/abs/2309.13957
"""
import torch
import numpy as np

import utils.chemistry_utils as chemistry_utils
from goal_directed_generation.utils import sample_unique_sequences

from oracles.oracle import Oracle
from goal_directed_generation.dataclass import GoalDirectedGenerationConfiguration
from models.model import Model
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
        oracle: Oracle,
        configuration: GoalDirectedGenerationConfiguration
    ):
        self.prior = Model.load_from_file(configuration.reinforcement_learning.prior)
        # Prior model is not updated so disable gradients
        self._disable_prior_gradients()
        self.agent = Model.load_from_file(configuration.reinforcement_learning.agent)

        # Seed for documentation
        self.seed = configuration.seed

        # Oracle
        self.oracle = oracle

        # RL parameters
        self.batch_size = configuration.reinforcement_learning.batch_size
        self.learning_rate = configuration.reinforcement_learning.learning_rate
        self.margin_threshold = configuration.reinforcement_learning.margin_threshold
        self.sigma = configuration.reinforcement_learning.sigma
        self.augmented_memory = configuration.reinforcement_learning.augmented_memory
        self.augmentation_rounds = configuration.reinforcement_learning.augmentation_rounds
        self.selective_memory_purge = configuration.reinforcement_learning.selective_memory_purge

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(configuration.experience_replay)

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
                                                    
        # TODO: Potentially implement MarginGuard
        #       --> self.margin_guard = MarginGuard(self)

        # only the Agent is updated
        self.optimizer = torch.optim.Adam(self.agent.get_network_parameters(), lr=self.learning_rate)
  
    def run(self):
        # FIXME: could be dangerous in case of infinite loop
        while not self.oracle.budget_exceeded():
            # 1. Sample unique SMILES from the Agent
            seqs, smiles, sampled_agent_likelihood = sample_unique_sequences(self.agent, self.batch_size)

            # 2. (Optional) Beam Enumeration: Filter SMILES using the Beam Enumeration pool
            if (self.execute_beam_enumeration) and (len(self.beam_enumeration.pool) != 0):
                seqs, smiles, sampled_agent_likelihood = self.beam_enumeration.filter_batch(seqs, smiles, sampled_agent_likelihood)

            # 3. (Optional) Beam Enumeration: If all SMILES are filtered, proceed to next generation epoch
            if len(smiles) == 0:
                self.beam_enumeration.filtered_epoch_updates()
                if self.beam_enumeration.patience_limit_reached():
                    print("Beam Enumeration: Patience limit reached. Ending.")
                    break
                continue

            # 4. Oracle call on sampled batch
            #    Rewards are already penalized by the Diversity Filter
            smiles, penalized_rewards = self.oracle(smiles, self.diversity_filter)

            # 5. (Optional) Beam Enumeration: Check whether to execute Beam Enumeration
            #               NOTE: Beam Enumeration execution criterion is based only *sampled* batch
            if self.execute_beam_enumeration:
                self.beam_enumeration.epoch_updates(
                    agent=self.agent,
                    num_valid_smiles=len(smiles),
                    reward=penalized_rewards.mean(),
                    oracle_calls=self.oracle.calls
                )

            # 5. (Optional) Hallucinated Memory: Hallucinate new SMILES from the Replay Buffer
            if (self.execute_hallucinated_memory) and (len(self.replay_buffer.memory) == 0):
                hallucinated_smiles = self.hallucinator.hallucinate(self.replay_buffer.memory)
                # 6. (Optional) Hallucinated Memory: Oracle call on hallucinated batch
                hallucinated_smiles, hallucinated_penalized_rewards = self.oracle(hallucinated_smiles, self.diversity_filter)
                # 7. (Optional) Update the hallucation history
                # FIXME: track how many times the replay buffer is being populated and not just when the hallucinations are the best-so-far
                #self.hallucinator.epoch_updates(
                #    epoch=step,
                #    highest_buffer_reward=highest_buffer_reward,
                #    hallucinations=hallucinated_smiles,
                #    hallucination_rewards=hallucinated_penalized_rewards)
            else:
                hallucinated_smiles, hallucinated_penalized_rewards = [], []

            # 7. Concatenate sampled batch with hallucinated batch
            smiles = np.concatenate((smiles, hallucinated_smiles), 0)
            penalized_rewards = np.concatenate((penalized_rewards, hallucinated_penalized_rewards), 0)

            # 8. Compute the loss
            prior_likelihood = torch.cat([-self.prior.likelihood_smiles(smiles), -self.prior.likelihood_smiles(hallucinated_smiles)], 0)
            agent_likelihood = torch.cat([-sampled_agent_likelihood, -self.agent.likelihood_smiles(hallucinated_smiles)], 0)
            loss = self.compute_loss(prior_likelihood, agent_likelihood, penalized_rewards)

            # 9. Update Replay Buffer
            #    Likelihoods should be negative here
            self.replay_buffer.add(
                smiles=smiles, 
                rewards=penalized_rewards, 
                prior_likelihood=prior_likelihood,
                agent_likelihood=agent_likelihood
            )

            # 10. Add experience replay to the loss - this is done *after* updating the Replay Buffer so new best-so-far SMILES can be sampled
            er_smiles, er_rewards, er_prior_likelihood = self.replay_buffer.sample_memory()

            # 11. Compute the loss for the experience replay SMILES
            if len(er_smiles) > 0:
                er_agent_likelihood = -self.agent.likelihood_smiles(er_smiles)
                er_loss = self.compute_loss(er_prior_likelihood, er_agent_likelihood, er_rewards)
            
            # 12. Concatenate to get the total loss and backpropagate
            loss = torch.cat((loss, er_loss), 0)
            self.backpropagate(loss)

            # 13. (Optional) Augmented Memory
            if self.augmented_memory:
                # NOTE: *Highly* recommended that Selective Memory Purge is enabled
                #       All penalized scaffolds are removed from the Replay Buffer *before* executing Augmented Memory
                if self.selective_memory_purge:
                    self.replay_buffer.selective_memory_purge(smiles, penalized_rewards)
                for _ in range(self.augmentation_rounds):
                    # Get randomized SMILES for both the sampled and hallucinated SMILES
                    randomized_smiles = chemistry_utils.randomize_smiles_batch(smiles, self.prior)
                    # Compute the loss
                    prior_likelihood = -self._prior.likelihood_smiles(randomized_smiles)
                    agent_likelihood = -self._agent.likelihood_smiles(randomized_smiles)
                    loss = self.compute_loss(prior_likelihood, agent_likelihood, penalized_rewards)
                    # Augmented Memory: Key operation for sample efficiency
                    randomized_buffer_smiles, randomized_buffer_rewards, randomized_prior_likelihood = self.replay_buffer.augmented_memory_replay()
                    randomized_agent_likelihood = -self.agent.likelihood_smiles(randomized_buffer_smiles)
                    augmented_memory_loss = self.compute_loss(randomized_prior_likelihood, randomized_agent_likelihood, randomized_buffer_rewards)
                    loss = torch.cat((loss, augmented_memory_loss), 0)
                    self.backpropagate(loss)

        # write out hallucination history
        self.hallucinator.write_out_history()

        if self.execute_beam_enumeration:
            self.beam_enumeration.end_actions(oracle_calls=self.oracle_tracker.oracle_calls)

        self.oracle.write_out_oracle_history()
        self.oracle.write_out_repeat_history()

    def compute_loss(
        self, 
        prior_likelihood: torch.Tensor, 
        agent_likelihood: torch.Tensor,
        rewards: np.ndarray[float]
    ) -> torch.Tensor:
        """
        Compute the loss for the RL agent.
        Based on REINVENT's original loss function: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x
        """
        augmented_likelihood = prior_likelihood + self.sigma * rewards
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
        return loss

    def backpropagate(self, loss: torch.Tensor) -> None:
        loss = loss.mean()
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

    def _disable_prior_gradients(self):
        for param in self.prior.get_network_parameters():
            param.requires_grad = False
