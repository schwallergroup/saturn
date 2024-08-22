import os
import subprocess
import tempfile
import yaml
import shutil
import pandas as pd
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
import uuid  # For generating temporary file names
from concurrent.futures import ThreadPoolExecutor



class Syntheseus(OracleComponent):
    """
    Wrapper around Syntheseus which itself is a wrapper around various retrosynthesis models and search algorithms.

    References:
    1. https://arxiv.org/abs/2310.19796
    2. https://microsoft.github.io/syntheseus/stable/
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # Syntheseus environment path
        self.env_name = self.parameters.specific_parameters.get("env_name", None)
        assert self.env_name is not None, "Please provide the Conda environment name with AiZynthFinder installed."

        # Whether to optimize for path length
        self.optimize_path_length = self.parameters.specific_parameters.get("optimize_path_length", False)

        # Whether to parallelize Syntheseus execution
        self.parallelize = self.parameters.specific_parameters.get("parallelize", True) # Defaults to True
        self.max_workers = self.parameters.specific_parameters.get("max_workers", 4)  # Default to 4 workers

        # Reaction model
        self.reaction_model = self._parse_model_name(self.parameters.specific_parameters.get("reaction_model", None))

        # Load building blocks
        self.building_blocks_file = self.parameters.specific_parameters.get("building_blocks_file", None)
        assert self.building_blocks_file is not None, "Please provide the path to the building blocks file."

        # Search time limit
        self.time_limit_s = self.parameters.specific_parameters.get("time_limit_s", 180)  # Default to 3 minutes per molecule

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # Store the current Syntheseus output for saving
        self.syntheseus_output = pd.DataFrame()

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[int]:
        # Reset output storage
        self.syntheseus_output = pd.DataFrame()
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._parallelized_compute_property(smiles, oracle_calls) if self.parallelize else self._compute_property(smiles, oracle_calls)
    
    def _parallelized_compute_property(
        self, 
        smiles: np.ndarray[str],
        oracle_calls: int
    ) -> np.ndarray[int]:
        """
        Thread Parallelized execution of Syntheseus on the SMILES batch.
        """
        # 1. Chunk the SMILES into max_worker batches
        smiles_chunks = np.array_split(smiles, self.max_workers)
        # Ensure "str" type
        smiles_chunks = [np.array(chunk.tolist()) for chunk in smiles_chunks]  # List[np.ndarray[str]]

        # 2. Multi-threaded execution
        workers = self.max_workers if len(smiles) > self.max_workers else 1
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(self._compute_property, smiles_chunks, [oracle_calls] * len(smiles_chunks))
            results = list(results)
            # Flatten the results
            output = np.array([])
            for result in results:
                output = np.concatenate((output, result))

        assert len(output) == len(smiles), "Syntheseus output length mismatch."
        # 3. Save the output
        self.aizynth_output.to_csv(os.path.join(self.output_dir, f"aizynth_output_{oracle_calls}.csv"), index=False)

        return output
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str],
        oracle_calls: int
    ) -> np.ndarray[int]:
        """
        Execute Syntheseus on the SMILES batch.
        """
        # TODO:
        # 3. Then write the config.yml file
        # 4. Then parse the output by model string matching to get the correct directory
        # 5. Then clean up


        # 1. Make a temporary directory to store the SMILES and output results
        temp_dir = tempfile.mkdtemp()

        # 2. Write the SMILES to the temporary directory
        with open(os.path.join(temp_dir, "smiles.smi"), "w") as f:
            # Syntheseus does not accept empty lines
            for idx, s in enumerate(smiles):
                if idx < len(smiles) - 1:
                    f.write(f"{s}\n")
                else:
                    f.write(f"{s}")

        # 3. Write the config.yml to the temporary directory
        self._write_config(temp_dir)

        # 4. Run Syntheseus
        output = subprocess.run([
            "conda",
            "run",
            "-n",
            self.env_name,
            "search",
            "--config", os.path.join(temp_dir, "config.yml")
        ], capture_output=True)

        # 5. Parse the output
        try:
            df = pd.read_json(output_file, orient="table")
            is_solved = np.array(df["is_solved"]).astype(int)
            # If solved, extract the number_of_steps - otherwise, set to 99
            # This is safe because one would always want to minimize the number of steps
            steps = [steps if solved else 99 for steps, solved in zip(df["number_of_steps"], df["is_solved"])]
            # Concatenate the output
            self.aizynth_output = pd.concat([self.aizynth_output, df], ignore_index=True)
            # In case AiZynthFinder is not being parallelized, directly save output
            if not self.parallelize:
                assert len(self.aizynth_output) == len(smiles), "AiZynthFinder output length mismatch."
                self.aizynth_output.to_csv(os.path.join(self.output_dir, f"aizynth_output_{oracle_calls}.csv"), index=False)
        except Exception as e:
            print(f"Error in parsing AiZynthFinder output: {e}")
            is_solved = np.zeros(len(smiles))
            steps = np.zeros(len(smiles))
            steps.fill(99)

        # 4. Delete the temporary folder and AiZynthFinder output
        shutil.rmtree(temp_dir)

        # 5. Prepare and/or return the output
        if self.optimize_path_length:
            return np.array(steps)
        else:
            # Even if the path length is not being optimized, the output is still the number of steps so that this information is tracked 
            # HACK: Path length is only meaningful if a route is solved. If not solved, set the path length = -99 
            #       to work with the "binary" Reward Shaping function which sets reward = 1 if path >= 1, 0 otherwise
            return np.array([steps if solved else -99 for steps, solved in zip(steps, is_solved)])

    def _write_config(self, dir_path: str) -> None:
        """
        Syntheseus can take as input a yaml file for easy execution. Write this yaml file.
        """
        # TODO: Can expose more Syntheseus parameters to the user in the future
        config = {
            "inventory_smiles_file": self.building_blocks_file,
            "search_targets_file": os.path.join(dir_path, "smiles.smi"),
            "model_class": self.reaction_model,
            'time_limit_s': self.time_limit_s,
            # Only return 1 result for now - this implies that if 1 solution is found, Syntheseus stops
            # NOTE: In retrosynthesis, it can be very useful to return multiple routes (unless a model's top-1 accuracy is 100%) 
            #       but we do this for simplicity at the moment, i.e., so long as *a* route is found, consider the molecule solved
            "num_top_results": 1,
            "results_dir": dir_path,
            # Saves storage memory
            "save_graph": False
        }

        # Write the data to the YAML file
        with open(os.path.join(dir_path, "config.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    @staticmethod
    def _parse_model_name(model_name: str) -> str:
        """
        Syntheseus expects proper capitalization of the model names. Parse user input and return the correct model name.
        """
        assert model_name is not None, "Please provide the reaction model name."
        if model_name in ["retroknn", "RetroKNN"]:
            return "RetroKNN"
        elif model_name in ["rootaligned", "RootAligned"]:
            return "RootAligned"
        elif model_name in ["graph2edits", "Graph2Edits"]:
            return "Graph2Edits"
        else:
            raise ValueError(f"Model name {model_name} not recognized or not supported yet.")
