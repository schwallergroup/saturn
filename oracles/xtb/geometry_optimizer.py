from typing import Tuple, List
import os
import tempfile
import shutil
import subprocess
from rdkit.Chem import Mol
from rdkit import Chem


class GeometryOptimizer:
    def __init__(self):
        # TODO: can include xTB specific parameters here
        pass
    def optimize_geometry(
        self, 
        mol: Mol
    ) -> Tuple[str, str, List[str]]:
        """
        Optimize geometry using OpenBabel then xTB (extreme accuracy).

        Returns:
            1. the path to the temp directory containing the optimized geometry and the xTB output
            2. the path to the optimized geometry
            3. the xTB output
        """
        smiles = Chem.MolToSmiles(mol)
        # Make temp folder to generate and store optimized geometries
        temp_dir = tempfile.mkdtemp()

        # Generate un-optimized geometry with OpenBabel
        openbabel_path = os.path.join(temp_dir, "openbabel.xyz")
        obabel_output = subprocess.run([
            "obabel", 
            f"-:{smiles}", 
            "--gen3d", 
            "-O", openbabel_path
        ], capture_output=True)

        # Call xTB to optimize geometry
        xtb_output = subprocess.run([
            "xtb", 
            openbabel_path, 
            "--opt", "[extreme]", 
            "--namespace", f"{temp_dir}/temp"
        ], capture_output=True)

        return temp_dir, os.path.join(temp_dir, "temp.xtbopt.xyz"), xtb_output.stdout.decode().splitlines()
    
    @staticmethod
    def clean_up_temp_dir(path: str):
        shutil.rmtree(path)
