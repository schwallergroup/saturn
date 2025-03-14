import os
import logging
import time

import torch

from oracles.oracle import Oracle
from scoring.dataclass import ScoringConfiguration

from diversity_filter.diversity_filter import DiversityFilter
from diversity_filter.dataclass import DiversityFilterParameters

from models.generator import Generator
from goal_directed_generation.utils import sample_unique_sequences

from utils.utils import setup_logging
from utils.chemistry_utils import remove_molecules_with_radicals, canonicalize_smiles_batch



class Scorer:
    """
    Runs the Oracle on a provided set of SMILES or samples SMILES from an Agent checkpoint and returns:

        1. Raw scores
        2. Transformed scores
        3. Negative log-likelihoods

    """
    def __init__(
        self, 
        logging_path: str,
        oracle: Oracle,
        diversity_filter_configuration: DiversityFilterParameters,
        configuration: ScoringConfiguration,
    ):
        # Oracle
        self.oracle = oracle

        # Diversity Filter
        self.diversity_filter = DiversityFilter(diversity_filter_configuration)

        # Set up logging
        self.logging_path = logging_path
        setup_logging(logging_path)

        # Sampling parameters
        self.smiles_path = configuration.smiles_path
        self.sample = configuration.sample
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = Generator.load_from_file(configuration.agent, device)
        self.sample_num = configuration.sample_num
        
        self.output_csv_path = configuration.output_csv_path

        if not self.sample:
            assert os.path.exists(self.smiles_path), f"SMILES file to compute scores is not found: {self.smiles_path} is invalid."
        # Make the output directory if it does not exist
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
  
    def run(self):
        start_time = time.perf_counter()

        # 1. Read the SMILES file or sample SMILES from the Agent checkpoint
        if not self.sample:
            smiles = []
            with open(self.smiles_path, "r") as f:
                for line in f.readlines():
                    smiles.append(line.strip())
        else:
            logging.info(f"Sampling {self.sample_num} SMILES from the Agent checkpoint")
            smiles = set()
            negative_log_likelihoods = []
            while len(smiles) < self.sample_num:
                _, sampled_smiles, _ = sample_unique_sequences(
                    agent=self.agent,
                    batch_size=512
                )
                # Remove potential radicals and canonicalize SMILES
                sampled_smiles = remove_molecules_with_radicals(sampled_smiles)
                canonical_smiles = set(canonicalize_smiles_batch(sampled_smiles))
                smiles.update(canonical_smiles)

                # Compute negative log-likelihoods
                nlls = self.agent.likelihood_smiles(sampled_smiles)
                negative_log_likelihoods.extend(nlls.detach().cpu().numpy())
            
            # Slice to get the desired number of SMILES
            smiles = list(smiles)[:self.sample_num]
            negative_log_likelihoods = negative_log_likelihoods[:self.sample_num]

        # 2. Oracle call on SMILES
        logging.info(f"Scoring {len(smiles)} SMILES...")
        scoring_start_time = time.perf_counter()
        smiles, _ = self.oracle(
            smiles=smiles,
            diversity_filter=self.diversity_filter
        )
        scoring_end_time = time.perf_counter()
        logging.info(f"Finished scoring in {round(scoring_end_time - scoring_start_time, 2)} seconds.")

        # 3. Write out results
        self.oracle.oracle_history["negative_log_likelihood"] = negative_log_likelihoods
        self.oracle.oracle_history.to_csv(self.output_csv_path, index=False)
 
        
        end_time = time.perf_counter()
        logging.info(f"Total wall time: {round(end_time - start_time, 2)} seconds.")
   