"""
Some code is based on the implementation from https://github.com/MolecularAI/Reinvent
"""

import time

import numpy as np
import torch
from reinvent_chemistry.utils import get_indices_of_unique_smiles
from reinvent_models.lib_invent.enums.generative_model_regime import GenerativeModelRegimeEnum
from reinvent_models.model_factory.configurations.model_configuration import ModelConfiguration
from reinvent_models.model_factory.enums.model_type_enum import ModelTypeEnum
from reinvent_models.model_factory.generative_model import GenerativeModel
from reinvent_models.model_factory.generative_model_base import GenerativeModelBase
from reinvent_scoring import FinalSummary
from reinvent_scoring.scoring.diversity_filters.reinvent_core.base_diversity_filter import BaseDiversityFilter
from reinvent_scoring.scoring.function.base_scoring_function import BaseScoringFunction

from running_modes.configurations import ReinforcementLearningConfiguration, BeamEnumerationConfiguration, HallucinatedMemoryConfiguration
from running_modes.constructors.base_running_mode import BaseRunningMode
from running_modes.reinforcement_learning.inception import Inception
from running_modes.reinforcement_learning.logging.base_reinforcement_logger import BaseReinforcementLogger
from running_modes.reinforcement_learning.margin_guard import MarginGuard
from running_modes.utils.general import to_tensor

from reinvent_chemistry.conversions import Conversions
from copy import deepcopy

from running_modes.beam_enumeration.beam_enumeration import BeamEnumeration
from running_modes.beam_enumeration.oracle_tracker import OracleTracker

from running_modes.hallucination.utils import initialize_hallucinator


class CoreReinforcementRunner(BaseRunningMode):

    def __init__(self, critic: GenerativeModelBase, actor: GenerativeModelBase,
                 configuration: ReinforcementLearningConfiguration,
                 beam_configuration: BeamEnumerationConfiguration, 
                 hallucinate_configuration: HallucinatedMemoryConfiguration,
                 scoring_function: BaseScoringFunction, diversity_filter: BaseDiversityFilter,
                 inception: Inception, logger: BaseReinforcementLogger):
        self._prior = critic
        self._agent = actor
        self._scoring_function = scoring_function
        self._diversity_filter = diversity_filter
        self.config = configuration
        self.beam_config = beam_configuration
        self.hallucination_config = hallucinate_configuration
        self._logger = logger
        self._inception = inception
        self._margin_guard = MarginGuard(self)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)

        # optimization algorithm
        self.optimization_algorithm = configuration.optimization_algorithm.lower()
        # specific algorithm parameters
        self.top_k = configuration.specific_algorithm_parameters.get("top_k", 0.5)
        self.alpha = configuration.specific_algorithm_parameters.get("alpha", 0.5)
        self.update_frequency = configuration.specific_algorithm_parameters.get("update_frequency", 5)
        # SMILES augmentation hyperparameters
        self.augmented_memory = configuration.augmented_memory
        self.augmentation_rounds = configuration.augmentation_rounds
        self.selective_memory_purge = configuration.selective_memory_purge
        # SMILES randomization functions from reinvent-chemistry
        self._chemistry = Conversions()

        # Oracle Tracker
        self.oracle_tracker = OracleTracker(oracle_limit=configuration.oracle_limit)

        # Beam Enumeration
        self.execute_beam_enumeration = self.beam_config.execute_beam_enumeration
        self.beam_enumeration = BeamEnumeration(k=self.beam_config.beam_k,
                                                beam_steps=self.beam_config.beam_steps,
                                                substructure_type=self.beam_config.substructure_type.lower(),
                                                substructure_min_size=self.beam_config.structure_min_size,
                                                pool_size=self.beam_config.pool_size,
                                                pool_saving_frequency=self.beam_config.pool_saving_frequency,
                                                patience=self.beam_config.patience,
                                                token_sampling_method=self.beam_config.token_sampling_method,
                                                filter_patience_limit=self.beam_config.filter_patience_limit)
        
        # Hallucinated Memory
        self.execute_hallucinated_memory = self.hallucination_config.execute_hallucinated_memory
        self.hallucinator = initialize_hallucinator(prior=self._prior,
                                                    hallucination_config=self.hallucination_config)
        
    def run(self):
        self._logger.log_message("starting an RL run")
        start_time = time.time()
        self._disable_prior_gradients()

        if (self.optimization_algorithm == "augmented_memory") or (self.optimization_algorithm == "reinvent"):
            for step in range(self.config.n_steps):

                if self.oracle_tracker.budget_exceeded():
                    if self.execute_beam_enumeration:
                        self.beam_enumeration.end_actions(oracle_calls=self.oracle_tracker.oracle_calls)
                    break

                seqs, smiles, agent_likelihood = self._sample_unique_sequences(self._agent, self.config.batch_size)
                # substructure filtering
                if (self.execute_beam_enumeration) and len(self.beam_enumeration.pool) != 0:
                    seqs, smiles, agent_likelihood = self.beam_enumeration.filter_batch(seqs, smiles, agent_likelihood)

                # in case no SMILES in the sampled batch contain the Beam substructures
                if len(smiles) == 0:
                    self.beam_enumeration.filtered_epoch_updates()
                    if self.beam_enumeration.patience_limit_reached():
                        print(f'----- Low probability substructures: ending run based on user-specified patience limit: {self.beam_enumeration.filter_patience_limit} NOTE: this is not an indication of the experiment failing. -----')
                        break
                    continue

                # switch signs
                agent_likelihood = -agent_likelihood
                prior_likelihood = -self._prior.likelihood(seqs)

                score_summary: FinalSummary = self._scoring_function.get_final_score_for_step(smiles, step)
                score = self._diversity_filter.update_score(score_summary, step)
                
                
                self.oracle_tracker.epoch_updates(num_valid_smiles=len(score_summary.valid_idxs),
                                                  epoch=step,
                                                  reward=score,
                                                  smiles=smiles)
                
                if self.execute_beam_enumeration:
                    self.beam_enumeration.epoch_updates(agent=self._agent,
                                                        num_valid_smiles=len(score_summary.valid_idxs),
                                                        reward=score.mean(),
                                                        oracle_calls=self.oracle_tracker.oracle_calls)
                    

                # ------------------------------------------------------------------------------------------------------------
                # hallucinate before first instance of experience replay in case the hallucinated molecules are better than the current best in the buffer
                # skip step 1 when buffer is empty
                if step != 0:
                    # 1. Hallucinate new SMILES
                    hallucinated_smiles = self.hallucinator.hallucinate(buffer=self._inception.memory)

                    # 2. Score the hallucinated SMILES
                    hallucinated_score_summary = self._scoring_function.get_final_score_for_step(hallucinated_smiles, step)
                    hallucinated_score = self._diversity_filter.update_score(hallucinated_score_summary, step)

                    # 3. Update oracle tracker to account for oracle calls from hallucinated molecules
                    self.oracle_tracker.epoch_updates(num_valid_smiles=len(hallucinated_score_summary.valid_idxs),
                                                      epoch=step,
                                                      reward=hallucinated_score,
                                                      smiles=hallucinated_smiles)

                    # 3. Update the replay buffer
                    # calculate the prior and agent likelihood of hallucinated SMILES
                    hallucinated_prior_likelihood = -self._prior.likelihood_smiles(hallucinated_smiles)
                    hallucinated_agent_likelihood = -self._agent.likelihood_smiles(hallucinated_smiles)
                    # add the hallucinated SMILES to the replay buffer (which importantly also purges the buffer to the buffer size)
                    # before updatunig the buffer, extract the current highest reward in the buffer
                    highest_buffer_reward = self._inception.memory.iloc[0]["score"]            
                    self._inception.add(smiles=hallucinated_smiles,
                                        score=hallucinated_score,
                                        neg_likelihood=hallucinated_prior_likelihood)
                    
                    # 4. Update the hallucation history
                    self.hallucinator.epoch_updates(epoch=step,
                                                    highest_buffer_reward=highest_buffer_reward,
                                                    hallucinations=hallucinated_smiles,
                                                    hallucination_rewards=hallucinated_score)
                    
                    
                    # concatenate sampled batch with hallucinated batch
                    smiles = np.concatenate((smiles, hallucinated_smiles), 0)
                    prior_likelihood = torch.cat((prior_likelihood, hallucinated_prior_likelihood), 0)
                    agent_likelihood = torch.cat((agent_likelihood, hallucinated_agent_likelihood), 0)
                    score = np.concatenate((score, hallucinated_score), 0)
                # ------------------------------------------------------------------------------------------------------------

                augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                # if augmented_memory is true, over-ride it here to not use it as we want to perform memory *after* sampling new SMILES in case of new "best-so-far" SMILES
                loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood, smiles, score, self._prior, override=True)

                loss = loss.mean()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self.augmented_memory:
                    if self.selective_memory_purge:
                        self._inception.selective_memory_purge(smiles, score)
                    for _ in range(self.augmentation_rounds):
                        # get randomized SMILES
                        randomized_smiles_list = self._chemistry.get_randomized_smiles(smiles, self._prior)
                        # get prior likelihood of randomized SMILES
                        prior_likelihood = -self._prior.likelihood_smiles(randomized_smiles_list)
                        # get agent likelihood of randomized SMILES
                        agent_likelihood = -self._agent.likelihood_smiles(randomized_smiles_list)
                        # compute augmented likelihood with the "new" prior likelihood using randomized SMILES
                        augmented_likelihood = prior_likelihood + self.config.sigma * to_tensor(score)
                        # compute loss
                        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
                        # experience replay using randomized SMILES
                        loss, agent_likelihood = self._inception_filter(self._agent, loss, agent_likelihood, prior_likelihood, randomized_smiles_list, score, self._prior)
                        loss = loss.mean()
                        self._optimizer.zero_grad()
                        loss.backward()
                        self._optimizer.step()

                self._stats_and_chekpoint(score, start_time, step, smiles, score_summary,
                                          agent_likelihood, prior_likelihood,
                                          augmented_likelihood)

            self._logger.save_final_state(self._agent, self._diversity_filter)
            self._logger.log_out_input_configuration()
            self._logger.log_out_inception(self._inception)
            # write out hallucination history
            self.hallucinator.write_out_history()

    def _disable_prior_gradients(self):
        # There might be a more elegant way of disabling gradients
        for param in self._prior.get_network_parameters():
            param.requires_grad = False

    def _stats_and_chekpoint(self, score, start_time, step, smiles, score_summary: FinalSummary,
                             agent_likelihood, prior_likelihood, augmented_likelihood):
        self._margin_guard.adjust_margin(step)
        mean_score = np.mean(score)
        self._margin_guard.store_run_stats(agent_likelihood, prior_likelihood, augmented_likelihood, score)
        self._logger.timestep_report(start_time, self.config.n_steps, step, smiles,
                                     mean_score, score_summary, score,
                                     agent_likelihood, prior_likelihood, augmented_likelihood, self._diversity_filter)
        self._logger.save_checkpoint(step, self._diversity_filter, self._agent)

    def _sample_unique_sequences(self, agent, batch_size):
        seqs, smiles, agent_likelihood = agent.sample(batch_size)
        unique_idxs = get_indices_of_unique_smiles(smiles)
        seqs_unique = seqs[unique_idxs]
        smiles_np = np.array(smiles)
        smiles_unique = smiles_np[unique_idxs]
        agent_likelihood_unique = agent_likelihood[unique_idxs]
        return seqs_unique, smiles_unique, agent_likelihood_unique

    def _inception_filter(self, agent, loss, agent_likelihood, prior_likelihood, smiles, score, prior, override=False):
        if self.augmented_memory and not override:
            if self._inception.configuration.augmented_memory_mode_collapse_guard:
                # if the below executes, Augmented Memory is effectively paused for this epoch
                self._inception.mode_collapse_guard()
            exp_smiles, exp_scores, exp_prior_likelihood = self._inception.augmented_memory_replay(prior)
        else:
            exp_smiles, exp_scores, exp_prior_likelihood = self._inception.sample()
        if len(exp_smiles) > 0:
            exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
            exp_augmented_likelihood = exp_prior_likelihood + self.config.sigma * exp_scores
            exp_loss = torch.pow((to_tensor(exp_augmented_likelihood) - exp_agent_likelihood), 2)

            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        self._inception.add(smiles, score, prior_likelihood)

        return loss, agent_likelihood

    def reset(self, reset_countdown=0):
        model_type_enum = ModelTypeEnum()
        model_regime = GenerativeModelRegimeEnum()
        actor_config = ModelConfiguration(model_type_enum.DEFAULT, model_regime.TRAINING,
                                          self.config.agent)
        self._agent = GenerativeModel(actor_config)
        self._optimizer = torch.optim.Adam(self._agent.get_network_parameters(), lr=self.config.learning_rate)
        self._logger.log_message("Resetting Agent")
        self._logger.log_message(f"Adjusting sigma to: {self.config.sigma}")
        return reset_countdown
