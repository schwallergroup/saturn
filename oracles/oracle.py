from typing import List
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from oracles.oracle_component import OracleComponent
from oracles.oracle_dataclass import OracleComponentParameters, OracleConfiguration
from diversity_filter.diversity_filter import DiversityFilter

from oracles.utils import construct_oracle_component


class Oracle:
    def __init__(
        self, 
        oracle_configuration: OracleConfiguration
    ):
        self.oracle_configuration = oracle_configuration
        # construct the oracle function which can be composed of >1 individual oracles (multi-parameter optimization)
        self.oracle = self.construct_oracle(oracle_configuration["components"])
        self.oracle_weights = [oracle.weight for oracle in self.oracle]
        self.aggregator = oracle_configuration["aggregator"]

        # track oracle budget
        self.budget = oracle_configuration["budget"]
        self.calls = 0

        # cache dictionary to store the results of previous oracle calls
        self.cache = dict()
        self.calls = 0

        # oracle history to assess sample efficiency via Generative Yield and Oracle Burden metrics
        self.oracle_history = pd.DataFrame({
            "oracle_calls": [],
            "smiles": [],
            "reward": [],
            "penalized_reward": []
        })
        # add each oracle component's raw value and reward to the oracle history DataFrame
        for oracle in self.oracle:
            self.oracle_history[f"{oracle.name}_raw_values"] = []
            self.oracle_history[f"{oracle.name}_reward"] = []

        print(self.oracle_history)
        exit()
        # NOTE: assume no repeated oracle calls are allowed


    def __call__(
        self, 
        smiles: np.ndarray[str],
        diversity_filter: DiversityFilter
    ) -> np.ndarray[float]:
        """
        The Oracle is called at every generation epoch and performs the following:
            1. Calls each oracle component in the Oracle 
            2. Aggregates the oracle feedback into a single scalar reward
            3. Penalizes the reward based on the diversity filter
            4. Updates the Oracle History which tracks oracle calls, rewards, and penalized rewards
            5. Updates the Oracle Cache to store the results of previous oracle calls
        """
        pass
        # 1. Only keep the valid SMILES (RDKit parsable)
        smiles = [s for s in smiles if Chem.MolFromSmiles(s) is not None]
        # 2. Get the Mols
        mols = np.vectorize(Chem.MolFromSmiles)(smiles)
        # 3. Execute preliminary check (if applicable)
        mols = self.execute_preliminary_check(mols)
        # 4. 
    

    
        # TODO: construct oracle should return a list of OracleComponent objects. __call__ should iterate through each component and call it
        #       each return value should already be subjected to a transformation function in the OracleComponent class
        #       the __call__ method should just aggregate the rewards either by weighted sum or weighted mean
     
        # NOTE: DO NOT forget to assign the OracleComponent weight 
    
        # each OracleComponent has a calculate reward method --> run through all of them in the list and then aggregate
    
        # NOTE: check the preliminary_check flag!!!!
    
        # TODO: only increment the NEW SMILES - check this by canonicalization BUT make sure not to return the canonicalization for backpropagation
        #       because augmented SMILES can be useful for likelihood learning
    

        # TODO: apply diversity filter!!
        reward = np.ones((30, 10000))
        penalized_reward = diversity_filter.penalize_reward(smiles, reward)

        # TODO: add raw rewards to this
        self.update_oracle_history(
            num_valid_smiles=len(mols),
            smiles=smiles,
            reward=reward,
            penalized_reward=penalized_reward
        )

        # TODO: update the cache!!! 
        # ******
        # cache the penalized rewards!!!!!
        # ******
                                   
        
        
    def construct_oracle(self, oracle_components: List[OracleComponentParameters]) -> List[OracleComponent]:
        """
        Construct the oracle function.
        """
        oracle = []
        for component in oracle_components:
            # construct the OracleComponent
            oracle_component = construct_oracle_component(component)
            oracle.append(oracle_component)

        return oracle
    
    def execute_preliminary_check(self, mols: np.ndarray[Mol]) -> np.ndarray[Mol]:
        """
        Executes a preliminary check (if applicable). Each oracle component has a preliminary_check flag that can be set to True.
        Components set to True will be executed first to check that the molecule satisfies that component based on a reward threshold.
        If the molecule does not satisfy the threshold, it is removed from the sampled batch.
        """
        # FIXME: set a threshold for each component. If not using Step transformation, rewards are not necessarily 0
        THRESHOLD = 0.05
        preliminary_check_oracles = [oracle for oracle in self.oracle if oracle.preliminary_check]

        if len(preliminary_check_oracles) != 0:
            filtered_mols = []
            for mol in mols:
                for idx, oracle in enumerate(preliminary_check_oracles):
                    _, rewards = oracle.calculate_reward(mol)
                    if (rewards > THRESHOLD) and (idx == len(preliminary_check_oracles) - 1):
                        filtered_mols.append(mol)
        else:
            return mols

    
    def update_oracle_history(
        self, 
        num_valid_smiles: int,
        smiles: np.ndarray[str],
        reward: np.ndarray[float],
        penalized_reward: np.ndarray[float]
    ) -> None:
        """
        This method performs 2 updates on every epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls
        """
        self.calls += num_valid_smiles
        # track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "smiles": smiles,
                "reward": reward, 
                "penalized_reward": penalized_reward, 
            })
        
        self.oracle_history = pd.concat([self.oracle_history, df])

    def update_generative_history(
        self, 
        num_valid_smiles: int,
        smiles: np.ndarray[str],
        reward: np.ndarray[float]
    ) -> None:
        """
        This method performs 2 updates on every epoch:
        1. Increments the number of oracle calls so far
        2. Updates the Oracle History that tracks the generative sampling as a function of oracle calls
        """
        self.calls += num_valid_smiles
        # track generated SMILES + reward as a function of oracle calls
        df = pd.DataFrame({
                "oracle_calls": np.full_like(smiles, self.calls),
                "reward": reward, 
                "smiles": smiles
            })
        
        self.oracle_history = pd.concat([self.oracle_history, df])

    def oracle_budget_exceeded(self) -> bool:
        """
        Check if the oracle budget has been exceeded.
        """
        return self.calls >= self.budget

    def write_out_oracle_history(self):
        """
        Write out the oracle history as a CSV.
        """
        self.oracle_history.to_csv("oracle_history.csv")
