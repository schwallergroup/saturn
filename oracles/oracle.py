from typing import List
import os
import pandas as pd
import numpy as np


class Oracle:
    def __init__(
            self, 
            oracle_configuration
        ):
        self.oracle_configuration = oracle_configuration
        # cache dictionary to store the results of previous oracle calls
        self.cache = dict()


    def __call__(self, smiles_batch: np.array) -> np.array:
        """
        Args:
            smiles_batch: np.array of strings of smiles
        Returns:
            np.array of rewards (float) of the same length as smiles_batch
        """
        
