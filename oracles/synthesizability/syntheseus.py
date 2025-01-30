from typing import Dict, Union
import os
import subprocess
import tempfile
import json
import ast
import yaml
import shutil
import pandas as pd
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from utils.chemistry_utils import canonicalize_smiles, construct_morgan_fingerprints_batch_from_file
from oracles.synthesizability.utils.utils import match_stock, extract_functional_groups, get_node_reward
from concurrent.futures import ThreadPoolExecutor
from oracles.synthesizability.utils.CONSTANTS import DEFAULT_TANGO_WEIGHTS

from oracles.synthesizability.dataclass import EnforcedBuildingBlocksParameters, EnforcedReactionsParameters



class Syntheseus(OracleComponent):
    """
    Wrapper around Syntheseus which implements various retrosynthesis models and search algorithms.

    References:
    1. https://arxiv.org/abs/2310.19796
    2. https://microsoft.github.io/syntheseus/stable/
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # Syntheseus environment path
        self.syntheseus_env_name = self.parameters.specific_parameters.get("syntheseus_env_name", None)
        assert self.syntheseus_env_name is not None, "Please provide the Conda environment name with Syntheseus installed."

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
        # NOTE: The dataclass is only used for organization purpose - still initialize every parameter to guard against missing/invalid parameters
        self.enforced_building_blocks_parameters = EnforcedBuildingBlocksParameters(
            **self.parameters.specific_parameters.get("enforced_building_blocks", None)
        )
        if self.enforced_building_blocks_parameters.enforce_blocks:
            self.enforce_start = self.enforced_building_blocks_parameters.enforce_start
            # Enforced building blocks
            self.enforced_building_blocks_file = self.enforced_building_blocks_parameters.enforced_building_blocks_file             
            assert self.enforced_building_blocks_file is not None, "The run specifies to enforce building blocks, please provide the path to the file containing the building blocks to enforce."
            self.enforced_building_blocks_smiles = [line.strip() for line in open(self.enforced_building_blocks_file, "r").readlines()]
            self.enforced_building_blocks_mols = [Chem.MolFromSmiles(smiles) for smiles in self.enforced_building_blocks_smiles]
            self.enforced_building_blocks_fps = construct_morgan_fingerprints_batch_from_file(self.enforced_building_blocks_file)
            self.enforced_building_blocks_functional_groups = extract_functional_groups(
                [canonicalize_smiles(smiles.strip()) for smiles in open(self.enforced_building_blocks_file, "r").readlines()]
            )  # Dict[str, List[str]

            self.use_dense_reward = self.enforced_building_blocks_parameters.use_dense_reward
            # Reward type
            self.reward_type = self.enforced_building_blocks_parameters.reward_type
            if self.use_dense_reward:
                assert self.reward_type is not None, "Using Dense Reward for Syntheseus but no reward type was provided."
            self.tango_weights = self.enforced_building_blocks_parameters.tango_weights
            if any(weight < 0 or weight is None for weight in self.tango_weights.values()):
                self.tango_weights = DEFAULT_TANGO_WEIGHTS

        # Enforced reactions
        # NOTE: The dataclass is only used for organization purpose - still initialize every parameter to guard against missing/invalid parameters
        self.enforced_reactions_parameters = EnforcedReactionsParameters(
            **self.parameters.specific_parameters.get("enforced_reactions", None)
        )

        self.include_rxn_info = self.parameters.specific_parameters.get("include_rxn_info", False)
        
        if self.enforced_reactions_parameters.enforce_rxn_class_presence:
            self.enforce_all_reactions = self.enforced_reactions_parameters.enforce_all_reactions
            self.enforced_rxn_classes = self.enforced_reactions_parameters.enforced_rxn_classes
            assert self.enforced_rxn_classes is not None, "The run specifies to enforce reactions, please provide the reaction classes to enforce."

        # Avoid reaction classes
        self.avoid_rxn_classes = self.enforced_reactions_parameters.avoid_rxn_classes

        # -------------------------------------------------------------------
        # Scripts required to extract reaction nodes and reaction information
        # -------------------------------------------------------------------

        # FIXME: Do not necessarily need these attributes - they are always initialized at the moment just for safety, fix later
        self.rxn_insight_env_name = self.enforced_reactions_parameters.rxn_insight_env_name
        assert self.rxn_insight_env_name is not None, "The run specifies to enforce reactions and/or include reaction information in the top graphs output, please provide the Conda environment name with Rxn-INSIGHT installed."

        self.rxn_insight_extraction_script_path = self.enforced_reactions_parameters.rxn_insight_extraction_script_path
        assert self.rxn_insight_extraction_script_path is not None, "The run requires extracting reaction information, please provide the path to the script that extracts the reaction classes from the Syntheseus route pickle file."

        if self.enforced_reactions_parameters.use_namerxn:
            self.namerxn_binary_path = self.enforced_reactions_parameters.namerxn_binary_path
            assert self.namerxn_binary_path is not None, "The run specifies to use NameRXN, please provide the path to the NameRXN executable. Note that this requires a license."

            self.namerxn_extraction_script_path = self.enforced_reactions_parameters.namerxn_extraction_script_path
            assert self.namerxn_extraction_script_path is not None, "The run specifies to use NameRXN, please provide the path to the NameRXN extraction script."

        # Path to the script that extracts the SMILES and depth from the Syntheseus route pickle file
        self.route_extraction_script_path = self.parameters.specific_parameters.get("route_extraction_script_path", None)
        # TODO: Avoid certain building blocks/reagents
        assert self.route_extraction_script_path is not None, "The run specifies to enforce/avoid building blocks and/or reactions, please provide the path to the script that extracts the SMILES and depth from the Syntheseus route pickle file."

        # Save top percentage routes
        self.save_top_routes = self.parameters.specific_parameters.get("save_top_routes", False)
        self.percentage_to_save = self.parameters.specific_parameters.get("percentage_to_save", 0.005)  # Default to top 0.5%
        assert self.percentage_to_save > 0.0 and self.percentage_to_save <= 1.0, "The save top percentage routes must be between 0.0 and 1.0."

        # Search time limit
        self.time_limit_s = self.parameters.specific_parameters.get("time_limit_s", 180)  # Default to 3 minutes per molecule

        # Output directory
        self.output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert self.output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(self.output_dir, exist_ok=True)

        # HACK: Save the SMILES order of the given batch to use as a mapping to serve Syntheseus route output
        #       This is necessary because if Syntheseus is parallelized, then the SMILES batch is chunked
        #       The order is lost as the count would start from 0 for each chunk
        self.smiles = None

        # Guard against invalid combination of parameters
        if (not self.enforced_building_blocks_parameters.enforce_blocks) and \
           (not self.enforced_reactions_parameters.enforce_rxn_class_presence) and \
           (len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0):
            assert self.parameters.reward_shaping_function_parameters["transformation_function"] == "binary", "The run specifies to enforce neither building blocks nor reaction classes, please use the Binary Reward Shaping function."

        # Trackers for matched SMILES under building block and reaction class constraints
        self.matched_generated_smiles = dict()
        self.matched_generated_smiles_with_rxn = dict()
        # Track the evolution of reaction classes/names (if applicable)
        self.smiles_to_rxn_tracker = dict()  # Dict[str, List[Tuple[int, str, str]]] --> {smiles: [(depth, rxn_class, rxn_name), ...]}

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
            self.syntheseus_env_name,
            "syntheseus",
            "search",
            "--config", os.path.join(temp_dir, "config.yml")
        ], capture_output=True)

        # 6. Parse the output
        is_solved = np.zeros(len(smiles))
        steps = np.zeros(len(smiles))
        steps.fill(99)
        node_rewards = np.zeros(len(smiles))
        try:
            # Syntheseus output is tagged by the reaction model name
            output_results_dir = [folder for folder in os.listdir(os.path.join(temp_dir)) if self.reaction_model in folder][0]

            # If there was more than 1 SMILES, Syntheseus outputs the results for each SMILES in a separate folder
            if len(smiles) > 1:
                output_files = [file for file in os.listdir(os.path.join(temp_dir, output_results_dir)) if not file.endswith(".json")]
                # *Important* to sort by ascending integer order so the molecules to output mapping is correct
                output_files = sorted(output_files, key=lambda x: int(x))

            # Otherwise, Syntheseus outputs the results directly in the output_results_dir folder
            else:
                output_files = [os.path.join(temp_dir, output_results_dir)]

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

                # Store the current generated SMILES being considered
                generated_smiles = canonicalize_smiles(smiles[idx])

                # ------------------------
                # ENFORCED BUILDING BLOCKS
                # ------------------------

                # If the molecule is solved *and* the user specified to enforce that a set of building blocks appears in the synthesis graph
                if self.enforced_building_blocks_parameters.enforce_blocks and int(is_solved[idx]) == 1:

                    if oracle_calls not in self.matched_generated_smiles:
                        self.matched_generated_smiles[oracle_calls] = []

                    route = self._extract_syntheseus_route_data(
                        route_path=os.path.join(temp_dir, output_results_dir, mol_results, "route_0.pkl"),
                        data_type="mol"
                    )

                    # Track the synthesis pathway contains an enforced building block
                    is_matched = False
                    
                    # Check whether to use dense reward
                    if self.use_dense_reward:
                        max_reward = 0.0
                        max_depth = self._get_max_depth(route)

                        # FIXME: Should be redundant as the generated SMILES is already tracked above
                        # Loop through the nodes in the route to ensure the root node (generated molecule) is tracked
                        for node, node_data in route.items():
                            # First extract the generated molecule
                            if node_data["depth"] == 0:
                                generated_smiles = canonicalize_smiles(node_data["smiles"])
                                break

                        # Loop through the nodes again, this time computing the reward for each node
                        for node, node_data in route.items():
                            # Skip root node because this is the generated molecule
                            if node_data["depth"] == 0:
                                continue
                            # If the user specified that enforced building blocks must appear in the nodes at max depth (starting-material)
                            if self.enforce_start:
                                if not node_data["depth"] == max_depth:
                                    continue
                            # Compute the node's reward
                            node_reward = get_node_reward(
                                reward_type=self.reward_type,
                                query_smiles=canonicalize_smiles(node_data["smiles"]),
                                enforce_blocks_fps=self.enforced_building_blocks_fps,
                                enforced_blocks_functional_groups=self.enforced_building_blocks_functional_groups,
                                tango_weights=self.tango_weights
                            )
                            max_reward = max(max_reward, node_reward)
                            # Check if the node exactly matches an enforced building block
                            is_matched, matched_block_smiles = match_stock(
                                query_smiles=canonicalize_smiles(node_data["smiles"]),
                                enforced_building_blocks_file=self.enforced_building_blocks_file
                            )
                            if is_matched:
                                self.matched_generated_smiles[oracle_calls].append(generated_smiles)
                                break
                    
                        node_rewards[idx] = max_reward

                        with open(os.path.join(self.output_dir, "matched_generated_smiles.json"), "w") as f:
                            json.dump(self.matched_generated_smiles, f, indent=4)
                    
                    # Otherwise, match *exactly*
                    else:
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
                            is_matched, matched_block_smiles = match_stock(
                                query_smiles=canonicalize_smiles(node_data["smiles"]), 
                                enforced_building_blocks_file=self.enforced_building_blocks_file
                            )
                            if is_matched:
                                break

                        is_solved[idx] = int(is_matched)
                        steps[idx] = steps[idx] if is_matched else 99

                # ------------------
                # ENFORCED REACTIONS
                # ------------------

                # Extract all reactions in the Syntheseus route if this information is required - either:
                #   1. Enforcing reaction classes
                #   2. Avoiding reaction classes
                if (self.enforced_reactions_parameters.enforce_rxn_class_presence and int(is_solved[idx]) == 1) or \
                   (self.enforced_reactions_parameters.avoid_rxn_classes and int(is_solved[idx]) == 1):

                    route = self._extract_syntheseus_route_data(
                        route_path=os.path.join(temp_dir, output_results_dir, mol_results, "route_0.pkl"),
                        data_type="rxn"
                    ) # The returned nodes are all Reaction nodes - extract reaction information from them

                    all_rxns = []  # List[Tuple[str, str]] --> (rxn_class, rxn_name)
            
                    # NameRXN classification of every reaction in the route
                    if self.enforced_reactions_parameters.use_namerxn:
                        reaction_depth_smiles = []  # [(depth, rxn_smiles), ...]
                        for node, node_data in route.items():
                            reaction_depth_smiles.append((node_data["depth"], node_data["rxn_smiles"]))

                        # Write the reaction SMILES to a temp fi.e
                        temp_rxn_smiles_file = os.path.join(temp_dir, "rxn_smiles.smi")
                        with open(temp_rxn_smiles_file, "w") as f:
                            for idx, (_, rxn_smiles) in enumerate(reaction_depth_smiles):
                                if idx < len(reaction_depth_smiles) - 1:
                                    f.write(f"{rxn_smiles}\n")
                                else:
                                    f.write(f"{rxn_smiles}")

                        all_rxns = subprocess.run([
                            "python",
                            self.namerxn_extraction_script_path,
                            self.namerxn_binary_path,
                            temp_rxn_smiles_file
                        ], capture_output=True, text=True)

                        # Check for errors
                        assert all_rxns.returncode == 0, f"Error during NameRXN reaction information extraction: {all_rxns.stderr}"
                        all_rxns = ast.literal_eval(all_rxns.stdout)

                    else:
                        for node, node_data in route.items():
                            # Execute Rxn-INSIGHT on the rxn SMILES
                            # HACK: This (temporary) solution enables reading the pickled data *without* installing Rxn-INSIGHT into the Saturn environment
                            extraction_result = subprocess.run([
                                "conda",
                                "run", 
                                "-n",
                                self.rxn_insight_env_name, 
                                "python", 
                                self.rxn_insight_extraction_script_path, 
                                # Pass the rxn SMILES extracted from the Syntheseus route
                                node_data["rxn_smiles"]
                            ], capture_output=True, text=True)

                            # Check for errors
                            assert extraction_result.returncode == 0, f"Error during Rxn-INSIGHT reaction information extraction: {extraction_result.stderr}"
                            rxn_info = ast.literal_eval(extraction_result.stdout)

                            all_rxns.append((rxn_info["CLASS"], rxn_info["NAME"]))

                # Assume the the reaction constraints are not satisfied
                rxn_multiplier = 0.0

                # If the molecule is solved *and* the user specified to enforce that a set of reaction classes appears in the synthesis graph
                if self.enforced_reactions_parameters.enforce_rxn_class_presence and int(is_solved[idx]) == 1:

                    if oracle_calls not in self.matched_generated_smiles_with_rxn:
                        self.matched_generated_smiles_with_rxn[oracle_calls] = []

                    for rxn_class, rxn_name in all_rxns:  # Un-pack each (rxn_class, rxn_name) pair
                        for enforced_rxn_class in self.enforced_rxn_classes:
                            # Convert to lower-case for more robust string comparison
                            if (enforced_rxn_class.lower() in rxn_class.lower()) or (enforced_rxn_class.lower() in rxn_name.lower()):
                                rxn_multiplier = 1.0
                            # If not enforcing all reactions, then finding *a* match is sufficient
                            if (rxn_multiplier == 1.0) and (not self.enforced_reactions_parameters.enforce_all_reactions):
                                break

                    # -----------------------------------------------------------------------------------------------
                    # NOTE: This block of code is only relevant when enforcing building blocks *and* reaction classes
                    # -----------------------------------------------------------------------------------------------

                    if not self.enforced_reactions_parameters.enforce_all_reactions:
                        # Check if the node exactly matches an enforced building block
                        if self.enforced_building_blocks_parameters.enforce_blocks: 
                            if (is_matched) and (rxn_multiplier) == 1.0 and len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0:
                                self.matched_generated_smiles_with_rxn[oracle_calls].append(generated_smiles)

                        elif not self.enforced_building_blocks_parameters.enforce_blocks:
                            # If the reaction class is matched, then the node reward is 1
                            if rxn_multiplier == 1.0 and len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0:
                                self.matched_generated_smiles_with_rxn[oracle_calls].append(generated_smiles)

                    # ------------------------------------------------------------------------
                    # NOTE: This block of code is only relevant when enforcing *all* reactions
                    # ------------------------------------------------------------------------

                    elif self.enforced_reactions_parameters.enforce_all_reactions:
                        for rxn_class, rxn_name in all_rxns:   # Un-pack each (rxn_class, rxn_name) pair
                            for enforced_rxn_class in self.enforced_rxn_classes:
                                if (enforced_rxn_class.lower() in rxn_class.lower()) or (enforced_rxn_class.lower() in rxn_name.lower()):
                                    break
                                else:
                                    rxn_multiplier = 0.0
                                    break
                        
                        # Enforcing blocks and *all* reactions
                        if self.enforced_building_blocks_parameters.enforce_blocks:
                            if (is_matched) and (rxn_multiplier) == 1.0 and len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0:
                                self.matched_generated_smiles_with_rxn[oracle_calls].append(generated_smiles)

                        # Only enforcing *all* reactions
                        else:
                            if rxn_multiplier == 1.0 and len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0:
                                self.matched_generated_smiles_with_rxn[oracle_calls].append(generated_smiles)

                    # Reaching this code requires that there is a solved route
                    # This truncates the node reward to 0 if the reaction class is not matched (assuming enforced blocks are also being considered)
                    # If *not* enforcing blocks, then the reward is 1.0 * rxn_multiplier because the route is solved in the first place and the user specified to enforce *only* reaction classes
                    node_rewards[idx] = node_rewards[idx] * rxn_multiplier if self.enforced_building_blocks_parameters.enforce_blocks else 1.0 * rxn_multiplier

                    # Write out the matched generated SMILES with reaction classes if not also avoiding reaction classes (otherwise, wait to check this)
                    if len(self.enforced_reactions_parameters.avoid_rxn_classes) == 0:
                        with open(os.path.join(self.output_dir, "matched_generated_smiles_with_rxn.json"), "w") as f:
                            json.dump(self.matched_generated_smiles_with_rxn, f, indent=4)          

                # -----------------------------------------------------------------------------------------------
                # NOTE: This block of code is only relevant when *avoiding* a set of reaction classes
                # -----------------------------------------------------------------------------------------------

                if len(self.enforced_reactions_parameters.avoid_rxn_classes) > 0 and int(is_solved[idx]) == 1:
                    avoid_rxn_multiplier = 1.0
                    # Check if specified reaction classes are *avoided*
                    for rxn_class, rxn_name in all_rxns:
                        for avoid_rxn_class in self.enforced_reactions_parameters.avoid_rxn_classes:
                            if (avoid_rxn_class.lower() in rxn_class.lower()) or (avoid_rxn_class.lower() in rxn_name.lower()):
                                avoid_rxn_multiplier = 0.0
                                break
                    
                    # In the scenario with enforcing reaction classes and the SMILES does not satisfy this constraint, the node_reward[idx] would already have been truncated to 0.
                    # Therefore, initializing another multiplier = 1.0 above would not change this outcome and is a safe operation
                    if self.enforced_building_blocks_parameters.enforce_blocks:
                        node_rewards[idx] = node_rewards[idx] * avoid_rxn_multiplier
                    else:
                        # Being very careful here: if the avoid_rxn_multiplier is 0.0, then the reward can automatically be set to 0.0
                        if avoid_rxn_multiplier == 0.0:
                            node_rewards[idx] = 0.0
                        # But if the avoid_rxn_multiplier is 1.0, then the reward might still be 0.0 if the user *also* wants to enforce reaction classes and this constraint is not satisfied
                        elif avoid_rxn_multiplier == 1.0:
                            if not self.enforced_reactions_parameters.enforce_rxn_class_presence:
                                node_rewards[idx] = 1.0
                            else:
                                node_rewards[idx] = node_rewards[idx] * 1.0

                    if node_rewards[idx] == 1.0:
                        self.matched_generated_smiles_with_rxn[oracle_calls].append(generated_smiles)
                        with open(os.path.join(self.output_dir, "matched_generated_smiles_with_rxn.json"), "w") as f:
                            json.dump(self.matched_generated_smiles_with_rxn, f, indent=4)          


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
            node_rewards = np.zeros(len(smiles))

        # 8. Delete the temporary directory and Syntheseus output
        shutil.rmtree(temp_dir)

        # 9. Prepare and/or return the output
        if self.enforced_building_blocks_parameters.enforce_blocks or \
           self.enforced_reactions_parameters.enforce_rxn_class_presence or \
           len(self.enforced_reactions_parameters.avoid_rxn_classes) > 0:

            if not self.parallelize:
                assert len(node_rewards) == len(smiles), "Syntheseus output length mismatch."

            return node_rewards

        elif self.optimize_path_length:
            if not self.parallelize:
                assert len(steps) == len(smiles), "Syntheseus output length mismatch."
            return steps

        else:
            # Even if the path length is not being optimized, the output is still the number of steps so that this information is tracked 
            # HACK: Path length is only meaningful if a route is solved. If not solved, set the path length = -99 
            #       to work with the "binary" Reward Shaping function which sets reward = 1 if path >= 1, 0 otherwise
            output = np.array([rxn_steps if solved else -99 for rxn_steps, solved in zip(steps, is_solved)], dtype=np.float32)
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

    def _write_out_top_synthesis_graphs(
        self,
        oracle_history: pd.DataFrame,
    ) -> bool:
        """
        Sort the Oracle History by reward and extract the corresponding Syntheseus synthesis graphs PDF files. 
        The purpose of this function is to automatically allow the user to visualize the synthesis routes for the top molecules.

        Returns:
            True if there are any solved molecules, False otherwise
        """
        # Output JSON with all relevant metrics and information
        output = {}
        
        # Keep only syntheseus_reward = 1, as these are the solved molecules
        oracle_history = oracle_history.loc[oracle_history["syntheseus_reward"] == 1]
        # If there are no solved molecules, return
        solved_exists = len(oracle_history) > 0
        if not solved_exists:
            return solved_exists
        
        # Sort the Oracle History by reward and extract the top percentage
        oracle_history = oracle_history.sort_values(by="reward", ascending=False)
        oracle_history = oracle_history.head(int(self.percentage_to_save * len(oracle_history)))
        
        # Loop through each top generated SMILES, extract the Syntheseus graph, and track which enforced smiles is visited (if applicable)
        enforced_blocks = []
        syntheseus_outputs = os.listdir(self.output_dir)
        pdf_paths = []

        for idx, (_, row) in enumerate(oracle_history.iterrows()):
            oracle_calls = int(row["oracle_calls"])
            generated_smiles = row["smiles"]
            # Extract the Oracle raw values
            reward = {
                "reward": row["reward"],
                **{col: row[col] for col in row.index if "raw_values" in col}
            }
            
            # Find the syntheseus output folder with the closest *smaller* number of oracle calls
            # FIXME: This is because currently, oracle calls is incremented before the Oracle History is updated. Fix this in the future. This is inelegant
            closest_smaller_oracle_calls_folder = max(
                [
                    folder for folder in syntheseus_outputs 
                    if not folder.endswith(".json") and 
                    not folder.endswith(".pdf") and 
                    not "graphs" in folder and
                    int(folder.split("_")[-1]) < oracle_calls
                ],
                key=lambda x: int(x.split("_")[-1])
            )

            # All the Mols matching the oracle calls
            mol_folder = os.listdir(os.path.join(self.output_dir, closest_smaller_oracle_calls_folder))
            # Loop through each to find the correct molecule
            for individual_mol_folder in mol_folder:
                added = False
                all_output_files = os.listdir(os.path.join(self.output_dir, closest_smaller_oracle_calls_folder, individual_mol_folder))
                # Check if the Mol is solved
                if "route_0.pkl" in all_output_files:
                    # Extract the route Mol data
                    route = self._extract_syntheseus_route_data(
                        route_path=os.path.join(self.output_dir, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
                        data_type="mol"
                    )
                    # Track which enforced block is visited (if applicable)
                    specific_enforced_block = None

                    for node, node_data in route.items():
                        # Check if the generated SMILES is in the route
                        if node_data["depth"] == 0 and canonicalize_smiles(node_data["smiles"]) == canonicalize_smiles(generated_smiles):
                            pdf_paths.append(os.path.join(self.output_dir, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pdf"))
                            added = True  
                            if self.enforced_building_blocks_parameters.enforce_blocks:
                                # Extract which enforced block is present
                                for intermediate_node, intermediate_node_data in route.items():
                                    canonical_intermediate_smiles = canonicalize_smiles(intermediate_node_data["smiles"])  # Canonicalize in case
                                    if canonical_intermediate_smiles in self.enforced_building_blocks_smiles:
                                        enforced_blocks.append(canonical_intermediate_smiles)
                                        specific_enforced_block = canonical_intermediate_smiles
                                        break
                            if added:
                                break
                    if added:
                        break

            # Get Synthesis data
            synthesis_data = self._extract_syntheseus_route_data(
                route_path=os.path.join(self.output_dir, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
                data_type="all"
            )     

            # For each Reaction node (is_rxn = True), extract the reaction information if the user specified to include reaction information
            if self.include_rxn_info:
                for node, node_data in synthesis_data.items():
                    if node_data["is_rxn"]:
                        extraction_result = subprocess.run([
                            "conda",
                            "run", 
                            "-n",
                            self.rxn_insight_env_name, 
                            "python", 
                            self.rxn_insight_extraction_script_path, 
                            # Pass the rxn SMILES extracted from the Syntheseus route
                            node_data["rxn_smiles"]
                        ], capture_output=True, text=True)

                        # Check for errors
                        assert extraction_result.returncode == 0, f"Error during Rxn-INSIGHT reaction information extraction: {extraction_result.stderr}"
                        rxn_info = ast.literal_eval(extraction_result.stdout)
                        node_data["rxn_class"] = rxn_info["CLASS"]
                        node_data["rxn_name"] = rxn_info["NAME"]

            # Construct the JSON for the current molecule
            output[generated_smiles] = {
                "reward": reward,
                "synthesis_data": synthesis_data,
                "enforced_block": specific_enforced_block
            }

        # Save the output JSON
        os.makedirs(os.path.join(self.output_dir, "top_synthesis_graphs"), exist_ok=True)
        with open(os.path.join(self.output_dir, "top_synthesis_graphs", "top_synthesis_graphs.json"), "w") as f:
            json.dump(output, f, indent=4)

        # Loop through each successful path and copy the "route_0.pdf" the syntheseus results directory
        for idx, path in enumerate(pdf_paths):
            os.system(f"cp {path} {os.path.join(self.output_dir, f'top_synthesis_graphs/route_{idx+1}.pdf')}")

        # Add the enforced blocks to the oracle history
        if self.enforced_building_blocks_parameters.enforce_blocks:
            try:
                oracle_history["enforced_blocks"] = enforced_blocks
            except Exception as e:
                print(f"Error adding enforced blocks to oracle history: {e}")
                oracle_history["enforced_blocks"] = ["nan"] * len(oracle_history)

        oracle_history.to_csv(os.path.join(self.output_dir, "top_synthesis_graphs", "top_synthesis_graphs.csv"), index=False)

        return solved_exists

    def _extract_syntheseus_route_data(
        self,
        route_path: str,
        data_type: str
    ) -> Dict[str, Union[str, int]]:
        # Read Syntheseus route data from the pickle file and extract the Mols or Reactions
        # HACK: This (temporary) solution enables reading the pickled data *without* installing Syntheseus into the Saturn environment
        extraction_result = subprocess.run([
            "conda", 
            "run", 
            "-n",
            self.syntheseus_env_name, 
            "python", 
            self.route_extraction_script_path, 
            # NOTE: We set Syntheseus to terminate after 1 route is found, so the index is always 0
            route_path,
            data_type,  # "mol", "rxn", or "all"
        ], capture_output=True, text=True)

        # Check for errors
        assert extraction_result.returncode == 0, f"Error during Syntheseus route ({data_type}) data extraction: {extraction_result.stderr}"
        route = json.loads(extraction_result.stdout)

        return route

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
        #       The default behaviour in Syntheseus is to download a trained model by the authors.
        #       These models are stored in .cache/torch/syntheseus by default.
        assert model_name is not None, "Please provide the reaction model name."
        if model_name in ["localretro", "LocalRetro"]:
            return "LocalRetro"
        elif model_name in ["retroknn", "RetroKNN"]:
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
