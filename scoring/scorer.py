import os
import logging
import time

from utils.utils import setup_logging

from oracles.oracle import Oracle
from scoring.dataclass import ScoringConfiguration

from diversity_filter.diversity_filter import DiversityFilter
from diversity_filter.dataclass import DiversityFilterParameters


class Scorer:
    """
    Runs the Oracle on a provided set of SMILES and returns the raw outputs and transformed rewards.
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

        self.smiles_path = configuration.smiles_path
        self.output_csv_path = configuration.output_csv_path

        assert os.path.exists(self.smiles_path), f"SMILES file not found: {self.smiles_path} is invalid."
        # Make the parent directory if it does not exist
        os.makedirs(os.path.dirname(self.output_csv_path), exist_ok=True)
        assert not os.path.exists(self.output_csv_path), f"Output CSV file path already exists: {self.output_csv_path}. Ending run to avoid overwriting."
  
    def run(self):
        start_time = time.perf_counter()

        # 1. Read SMILES
        smiles = []
        with open(self.smiles_path, "r") as f:
            for line in f.readlines():
                smiles.append(line.strip())

        # FIXME: Assumes all input SMILES are unique
        #        Technically, non-unique SMILES will not cause any errors but it does not make sense to score the same SMILES multiple times (unless interrogating oracle stochasticity).
        logging.info(f"Scoring {len(smiles)} input SMILES")

        # 2. Oracle call on SMILES
        smiles, _ = self.oracle(smiles, self.diversity_filter)

        # 3. Write out results
        self.oracle.oracle_history.to_csv(self.output_csv_path, index=False)
 
        logging.info(f"Finished scoring.")
        
        end_time = time.perf_counter()
        logging.info(f"Total wall time: {end_time - start_time} seconds.")
   