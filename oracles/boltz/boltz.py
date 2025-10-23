"""Based on Boltz-2 implementation: https://github.com/jwohlwend/boltz
"""
import numpy as np
from numpy import ndarray
from rdkit.Chem import Mol
from rdkit import Chem
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from concurrent.futures import ThreadPoolExecutor
from typing import Union
import tempfile
import subprocess
import os
import json
import shutil
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem



class Boltz(OracleComponent):
    """
    Boltz-2 is a protein structure and ligand affinity prediction model. It incorporates
    an affinity module to predict both the binding affinity and the probability of binding 
    to a protein of a specific ligand. Reference: https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # boltz environment path
        self.boltz_env_name = parameters.specific_parameters.get("boltz_env_name", None)
        assert self.boltz_env_name is not None, "Please provide the Conda environment name with boltz installed."

        # FASTA sequence of the target
        self.protein_sequence = parameters.specific_parameters.get("protein_sequence", None)
        assert self.protein_sequence is not None, "Please provide the protein FASTA sequence"

        # Precomputed MSA file of the target sequence
        self.msa_file_path = parameters.specific_parameters.get("msa_file_path", None)
        assert self.msa_file_path is not None, "Please provide a precomputed MSA file with a .csv format"

        # Saving folder for predictions
        self.output_dir = parameters.specific_parameters.get("saving_folder", None)
        assert self.output_dir is not None, "Please provide a path to save Boltz predictions"
        os.makedirs(self.output_dir, exist_ok=True)

        # Whether to parallelize Boltz execution (specify number of threads)
        self.parallelize = parameters.specific_parameters.get("parallelize", 1)

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:

        smiles = np.vectorize(Chem.MolToSmiles)(mols)

        scores = np.zeros(len(smiles))

        # 1. We need to check which molecules can be computed into a conformer to avoid Boltz error
        valid_molecules_indices = np.where(np.array([self._can_embed_conformer(smile) for smile in smiles]) == True)[0]
        
        # Only valid molecules are passed to Boltz
        valid_smiles = smiles[valid_molecules_indices]

        if self.parallelize == 1:
            valid_scores = self._compute_property(valid_smiles)
        
        else:
            valid_scores = self._parallelize_compute(valid_smiles, self.parallelize)

        scores[valid_molecules_indices] = valid_scores

        return scores

    def _compute_property(self,
                          smiles: np.ndarray[str]) -> np.ndarray[Union[int, float]]:
        """Run boltz to get affinity prediction.
        """
        scores = np.zeros(len(smiles))

        # Make temporary directory to store input and output files
        temp_dir = tempfile.mkdtemp()

        # Write input files using indexes
        self._write_input_yaml(temp_dir,
                               smiles)
        
        # Call boltz in a subprocess
        output = subprocess.run([
                "conda",
                "run",
                "-n",
                self.boltz_env_name,
                "boltz",
                "predict",
                os.path.join(temp_dir),
                "--seed",
                "0"
            ], capture_output=False)

        # Get the name of the last directory of temp_dir
        last_dir_name = os.path.basename(os.path.join(temp_dir))

        # Extract prediction for each molecule
        preds_path = os.path.join(f"boltz_results_{last_dir_name}", "predictions")

        folders = sorted(os.listdir(preds_path))

        # Reference index to save files based on the molecules that were already saved
        target_index = len(os.listdir(self.output_dir)) // 2

        for i, file in enumerate(folders):
            folder_path = os.path.join(preds_path, file)
            
            index = int(file.split("_")[-1])

            # Check if file path exists, otherwise score == 0
            file_path = f"{folder_path}/affinity_{file}.json"

            if not os.path.exists(file_path):
                scores[index] = 0
                continue

            with open(file_path, "r") as f:
                data = json.load(f)
                affinity = data["affinity_probability_binary"]
                scores[index] = affinity

            # Copy .cif and affinity file
            cif_path = f"{folder_path}/{file}_model_0.cif"

            # copy .cif
            subprocess.run([
                "cp",
                "-r",
                cif_path,
                os.path.join(self.output_dir, f"{target_index+i}.cif")
            ])
            
            # copy .json
            subprocess.run([
                "cp",
                "-r",
                file_path,
                os.path.join(self.output_dir, f"{target_index+i}.json")
            ])
        
        # Remove final folder
        shutil.rmtree(os.path.join(f"boltz_results_{last_dir_name}"))

        return scores

        
    def _parallelize_compute(self,
                             smiles: np.ndarray[str],
                             max_workers: int = 4) -> np.ndarray[Union[float, int]]:
        """Parallelize Boltz execution in the same GPU.
        FIXME: multithreading/multiprocessing do not provide significant acceleration,
        we should use multi-GPU instead.
        """

        # Chunk SMILES
        smiles_chunks = np.array_split(smiles, max_workers)

        # Ensure str type
        smiles_chunks = [np.array(chunk.tolist()) for chunk in smiles_chunks]

        # Multithread execution
        workers = max_workers if len(smiles) >= max_workers else 1
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(self._compute_property, smiles_chunks)
            results = list(results)

            # Flatten outputs
            output = np.array([])

            for result in results:
                output = np.concatenate((output, result))
        
        return output
    
    def _write_input_yaml(self,
                        dir_path: str,
                        smiles: np.ndarray) -> None:
        """Write input yaml file. If indices
        """

        for i, smile in enumerate(smiles):

            data = {
                "version": 1,
                "sequences": [
                    {"protein": {"id": "A", 
                                 "sequence": str(self.protein_sequence),
                                 "msa": str(self.msa_file_path)}},
                    {"ligand":  {"id": "B", "smiles":  str(smile)}},
                ],
                "properties": [
                    {"affinity": {"binder": "B"}}
                ],
            }

            with open(f"{dir_path}/input_{i}.yaml", "w") as f:
                yaml.safe_dump(
                    data, f,
                    sort_keys=False,          
                    default_flow_style=False,  
                    allow_unicode=True,
                )
    
    def _can_embed_conformer(self,
                             smiles: str) -> bool:
        """Return True if a 3D conformer can be generated from SMILES using RDKit ETKDGv3.
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return False
            mol = Chem.AddHs(mol)
            #  
            params = AllChem.ETKDGv3()
            cid = AllChem.EmbedMolecule(mol, params)
            return cid >= 0
        except Exception:
            return False