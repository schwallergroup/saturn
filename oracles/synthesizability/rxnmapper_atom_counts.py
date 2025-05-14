import os
import subprocess
import tempfile
import shutil
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol
from utils.chemistry_utils import canonicalize_smiles_batch



class RXNMapperAtomCounts(OracleComponent):
    """
    Count the fraction of atoms in the enforced structures that can be mapped to the generated molecules using RXNMapper.

    References:
    1. https://www.science.org/doi/10.1126/sciadv.abe4166
    2. https://github.com/rxn4chemistry/rxnmapper
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.env_name = self.parameters.specific_parameters.get("env_name", None)
        assert self.env_name is not None, "Environment name with rxnmapper installed must be provided."

        self.rxnmapper_script_path = self.parameters.specific_parameters.get("rxnmapper_script_path", None)
        assert self.rxnmapper_script_path is not None, "rxnmapper script path must be provided."

        self.enforced_structures = parameters.specific_parameters.get("enforced_structures")
        assert self.enforced_structures is not None, "Enforced Structures must be provided"
        assert type(self.enforced_structures) in [list, str], "Enforced Structures must be a list of SMILES or a path to a file containing SMILES"

        if type(self.enforced_structures) == list:
            self.enforced_structures_smiles = self.enforced_structures
        else:
            with open(self.enforced_structures, "r") as f:
                self.enforced_structures_smiles = [line.strip() for line in f.readlines()]

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        # Write the generated molecules and enforced structures to a temporary file
        temp_dir = tempfile.mkdtemp()
        with open(os.path.join(temp_dir, "generated_smiles.smi"), "w") as f:
            smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]
            for s in smiles:
                f.write(f"{s}\n")
        with open(os.path.join(temp_dir, "enforced_structures.smi"), "w") as f:
            smiles = canonicalize_smiles_batch(self.enforced_structures_smiles)
            for s in smiles:
                f.write(f"{s}\n")

        # HACK: This (temporary) solution enables running rxnmapper *without* installing it into the Saturn environment
        output = subprocess.run([
            "conda", 
            "run", 
            "-n",
            self.env_name, 
            "python", 
            self.rxnmapper_script_path, 
            # Pass generated *canonical* SMILES
            os.path.join(temp_dir, "generated_smiles.smi"), 
            # Pass enforced structures *canonical* SMILES
            os.path.join(temp_dir, "enforced_structures.smi")
        ], capture_output=True, text=True)

        # Check for errors
        assert output.returncode == 0, f"Error during rxnmapper execution: {output.stderr}"
        results = output.stdout
        
        # Parse results
        results = results.replace("[", "").replace("]", "").split()

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

        return np.array(results, dtype=np.float32)
