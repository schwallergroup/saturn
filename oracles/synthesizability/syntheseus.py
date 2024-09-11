from typing import Dict, Union
import os
import subprocess
import tempfile
import json
import yaml
import shutil
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from utils.chemistry_utils import canonicalize_smiles, construct_morgan_fingerprints_batch_from_file
from oracles.synthesizability.utils.utils import match_stock, get_max_stock_similarity, extract_functional_groups, matched_fuzzy_substructure, matched_functional_groups
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
        assert self.env_name is not None, "Please provide the Conda environment name with Syntheseus installed."

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

        # Enforced building blocks
        self.enforce_blocks = self.parameters.specific_parameters.get("enforce_blocks", False)  # Whether to enforce synthetic routes cross a set of reference blocks
        if self.enforce_blocks:
            # Path to the script that extracts the SMILES and depth from the Syntheseus route pickle file
            self.route_extraction_script_path = self.parameters.specific_parameters.get("route_extraction_script_path", None)
            assert self.route_extraction_script_path is not None, "The run specifies to enforce building blocks, please provide the path to the script that extracts the SMILES and depth from the Syntheseus route pickle file."

            self.enforced_building_blocks_file = self.parameters.specific_parameters.get("enforced_building_blocks_file", None)
            assert self.enforced_building_blocks_file is not None, "The run specifies to enforce building blocks, please provide the path to the building blocks file."

            self.enforce_start = self.parameters.specific_parameters.get("enforce_start", False)  # Whether to enforce that the reference blocks appear in the root nodes

            self.use_tanimoto_similarity = self.parameters.specific_parameters.get("use_tanimoto_similarity", True)  # Whether to use Tanimoto similarity as metric of "matching" to reference blocks
            if self.use_tanimoto_similarity:
                self.enforced_building_blocks_mols = [Chem.MolFromSmiles(line.strip()) for line in open(self.enforced_building_blocks_file, "r").readlines()]
                self.enforced_building_blocks_fps = construct_morgan_fingerprints_batch_from_file(self.enforced_building_blocks_file)
                self.enforced_building_blocks_functional_groups = extract_functional_groups(self.enforced_building_blocks_file)
                self.enforce_functional_groups = self.parameters.specific_parameters.get("enforce_functional_groups", True)
                self.enforce_fuzzy_substructure = self.parameters.specific_parameters.get("enforce_fuzzy_substructure", True) 

        # Search time limit
        self.time_limit_s = self.parameters.specific_parameters.get("time_limit_s", 180)  # Default to 3 minutes per molecule

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        # HACK: Save the SMILES order of the given batch to use as a mapping to serve Syntheseus route output
        #       This is necessary because if Syntheseus is parallelized, then the SMILES batch is chunked
        #       The order is lost as the count would start from 0 for each chunk
        self.smiles = None

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[int]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        # Save SMILES order - overwrites the previous batch's SMILES
        self.smiles = smiles
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
        return output
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str],
        oracle_calls: int
    ) -> np.ndarray[int]:
        """
        Execute Syntheseus on the SMILES batch.
        """
        # 1. Get the indices of the SMILES given the *generative model's* batch order
        save_indices = np.where(np.isin(self.smiles, smiles))[0]
        # Convenience for parsing so that the "first" molecule in the batch is at index 1 instead of 0
        save_indices += 1

        # 2. Make a temporary directory to store the SMILES and output results
        temp_dir = tempfile.mkdtemp()

        # 3. Write the SMILES to the temporary directory
        with open(os.path.join(temp_dir, "smiles.smi"), "w") as f:
            # Syntheseus does not accept empty lines
            for idx, s in enumerate(smiles):
                if idx < len(smiles) - 1:
                    f.write(f"{s}\n")
                else:
                    f.write(f"{s}")

        # 4. Write the config.yml to the temporary directory
        self._write_config(temp_dir)

        # 5. Run Syntheseus
        output = subprocess.run([
            "conda",
            "run",
            "-n",
            self.env_name,
            "syntheseus",
            "search",
            "--config", os.path.join(temp_dir, "config.yml")
        ], capture_output=True)

        # 6. Parse the output
        is_solved = np.zeros(len(smiles))
        steps = np.zeros(len(smiles))
        steps.fill(99)
        tan_sims = np.zeros(len(smiles))
        try:
            # Syntheseus output is tagged by the reaction model name
            output_results_dir = [folder for folder in os.listdir(os.path.join(temp_dir)) if self.reaction_model in folder][0]
            output_files = [file for file in os.listdir(os.path.join(temp_dir, output_results_dir)) if not file.endswith(".json")]
            # *Important* to sort by ascending integer order so the molecules to output mapping is correct
            output_files = sorted(output_files, key=lambda x: int(x))
            # Loop through the results for each query SMILES and extract the number of model calls 
            # required to solve. This is the number of reaction steps 
            for idx, mol_results in enumerate(output_files):
                # Load the JSON file
                with open(os.path.join(temp_dir, output_results_dir, mol_results, "stats.json"), "r") as f:
                    stats = json.load(f)
                # Extract the number of steps
                num_rxn_steps = stats["soln_time_rxn_model_calls"]
                is_solved[idx] = 1 if num_rxn_steps != np.inf else 0
                steps[idx] = int(num_rxn_steps) if num_rxn_steps != np.inf else 99

                # If the molecule is solved *and* the user specified to enforce that a set of building blocks appears in the synthesis graph
                if self.enforce_blocks and int(is_solved[idx]) == 1:
                    # Read the route data from the pickle file
                    # HACK: This (temporary) solution enables reading the pickled data *without* installing Syntheseus into the Saturn environment
                    extraction_result = subprocess.run([
                        "conda", 
                        "run", 
                        "-n",
                        self.env_name, 
                        "python", 
                        self.route_extraction_script_path, 
                        # NOTE: We set Syntheseus to terminate after 1 route is found, so the index is always 0
                        os.path.join(temp_dir, output_results_dir, mol_results, "route_0.pkl")
                    ], capture_output=True, text=True)

                    # Check for errors
                    assert extraction_result.returncode == 0, f"Error during Syntheseus route data extraction: {extraction_result.stderr}"
                    route = json.loads(extraction_result.stdout)

                    # Check if the generated molecule (at depth = 0) has a fuzzy substructure match to the enforced building blocks
                    if self.enforce_fuzzy_substructure:
                        for node, node_data in route.items():
                            if node_data["depth"] == 0:
                                if not matched_fuzzy_substructure(
                                    generated_smiles=canonicalize_smiles(node_data["smiles"]), 
                                    enforced_blocks=self.enforced_building_blocks_mols
                                ):
                                    is_solved[idx] = 0
                                    steps[idx] = 99
                                    break
                        continue

                    # Check whether to match by Tanimoto similarity
                    if self.use_tanimoto_similarity:
                        max_tan_sim = 0.0
                        max_depth = self._get_max_depth(route)
                        for node, node_data in route.items():
                            # Skip root node because this is the generated molecule
                            if node_data["depth"] == 0:
                                continue
                            # If the user specified that enforced building blocks must appear in the *leaf* nodes
                            if self.enforce_start:
                                # Skip nodes that are not at max depth (leaf nodes)
                                if not node_data["depth"] == max_depth:
                                    continue
                            # Check whether there is sufficient functional groups overlap between the query and enforced building blocks
                            if self.enforce_functional_groups:
                                if not matched_functional_groups(
                                    query_smiles=canonicalize_smiles(node_data["smiles"]), 
                                    enforced_blocks_functional_groups=self.enforced_building_blocks_functional_groups
                                ):
                                    continue

                            # At this point, depending on the user's specifications, the current node may be a combination of the following:
                            #   1. The generated molecule (root node) has a fuzzy substructure match to the enforced building blocks
                            #   2. The current node is a leaf node (depth = 0)
                            #   3. The current node has sufficient functional groups overlap with the enforced building blocks
                            max_tan_sim = max(max_tan_sim, get_max_stock_similarity(node_data["smiles"], self.enforced_building_blocks_fps))

                        tan_sims[idx] = max_tan_sim

                    # Otherwise, match *exactly*
                    else:
                        is_matched = False
                        max_depth = self._get_max_depth(route)
                        for node, node_data in route.items():
                            # Skip root node because this is the generated molecule
                            if node_data["depth"] == 0:
                                continue
                            # If the user specified that enforced building blocks must appear in the *leaf* nodes
                            if self.enforce_start:
                                if not node_data["depth"] == max_depth:
                                    continue
                            # Otherwise, just check if *any* node has a matching SMILES in the enforced building blocks file
                            # NOTE: this means that the enforced building blocks appear *somewhere* in the synthesis graph
                            is_matched = match_stock(
                                query_smiles=canonicalize_smiles(node_data["smiles"]), 
                                enforced_building_blocks_file=self.enforced_building_blocks_file
                            )
                            if is_matched:
                                break

                        is_solved[idx] = is_solved[idx] if is_matched else 0
                        steps[idx] = steps[idx] if is_matched else 99

            # HACK: In case a molecule is in the building blocks stock, Syntheseus returns 0. 
            #       Set these to 1 to work with Binary Reward Shaping
            steps[steps == 0] = 1

            # 7. Copy the output to the output directory
            for mol_results, save_index in zip(output_files, save_indices):
                # Make output folder tagged by the oracle calls
                os.makedirs(os.path.join(self.output_dir, f"output_{oracle_calls}"), exist_ok=True)
                subprocess.run([
                    "cp",
                    "-r",
                    os.path.join(temp_dir, output_results_dir, mol_results),
                    os.path.join(self.output_dir, f"output_{oracle_calls}", f"mol_{save_index}")
                ])
            
        except Exception as e:
            print(f"Error in parsing Syntheseus output: {e}")
            is_solved = np.zeros(len(smiles))
            steps = np.zeros(len(smiles))
            steps.fill(99)

        # 8. Delete the temporary directory and Syntheseus output
        shutil.rmtree(temp_dir)

        # 9. Prepare and/or return the output
        if self.use_tanimoto_similarity:
            if not self.parallelize:
                assert len(tan_sims) == len(smiles), "Syntheseus output length mismatch."
            return tan_sims
        elif self.optimize_path_length:
            if not self.parallelize:
                assert len(steps) == len(smiles), "Syntheseus output length mismatch."
            return steps
        else:
            # Even if the path length is not being optimized, the output is still the number of steps so that this information is tracked 
            # HACK: Path length is only meaningful if a route is solved. If not solved, set the path length = -99 
            #       to work with the "binary" Reward Shaping function which sets reward = 1 if path >= 1, 0 otherwise
            output = np.array([rxn_steps if solved else -99 for rxn_steps, solved in zip(steps, is_solved)])
            if not self.parallelize:
                assert len(output) == len(smiles), "Syntheseus output length mismatch."
            return output

    def _write_config(self, dir_path: str) -> None:
        """
        Syntheseus can take as input a yaml file for easy execution. Write this yaml file.
        """
        # TODO: Can expose more Syntheseus parameters to the user in the future
        config = {
            "inventory_smiles_file": self.building_blocks_file,
            "search_targets_file": os.path.join(dir_path, "smiles.smi"),
            "model_class": self.reaction_model,
            "time_limit_s": self.time_limit_s,
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
    def _get_max_depth(route: Dict[str, Union[str, int]]) -> int:
        """
        Get the maximum depth of the synthesis graph.
        """
        max_depth = 0
        for node, node_data in route.items():
            max_depth = max(max_depth, node_data["depth"])
        return max_depth

    @staticmethod
    def _parse_model_name(model_name: str) -> str:
        """
        Syntheseus expects proper capitalization of the model names. Parse user input and return the correct model name.
        """
        # NOTE: Assumes the user is not providing their own trained model.
        #       The default behaviour in Syntheseus is then to download a trained model by the authors.
        #       These models are stored in .cache/torch/syntheseus by default.
        assert model_name is not None, "Please provide the reaction model name."
        if model_name in ["retroknn", "RetroKNN"]:
            return "RetroKNN"
        elif model_name in ["rootaligned", "RootAligned"]:
            return "RootAligned"
        elif model_name in ["graph2edits", "Graph2Edits", "graph2edit", "Graph2Edit"]:
            return "Graph2Edits"
        elif model_name in ["megan", "MEGAN"]:
            return "MEGAN"
        else:
            # TODO: Support all the models in Syntheseus 
            raise ValueError(f"Model name {model_name} not recognized or not supported yet.")
