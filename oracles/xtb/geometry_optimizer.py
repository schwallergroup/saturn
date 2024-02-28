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
    def optimize_geometry(self, mol: Mol):
        """
        Optimize geometry using OpenBabel then xTB (extreme accuracy).
        Returns the path to the temp directory containing the optimized geometry.
        """
        smiles = Chem.MolToSmiles(mol)
        # Make temp folder to generate and store optimized geometries
        temp_dir = tempfile.mkdtemp()
        # Generate un-optimized geometry with OpenBabel
        openbabel_path = os.path.join(temp_dir, "openbabel.xyz")
        subprocess.run(["obabel", f"-:{smiles}", "--gen3d", "-O", openbabel_path])
        # Call xTB to optimize geometry
        subprocess.call(["xtb", openbabel_path, "--opt", "[extreme]", "--namespace", f"{temp_dir}/temp"])

        return temp_dir, os.path.join(temp_dir, "temp.xtbopt.xyz")
    
    @staticmethod
    def clean_up_temp_dir(path: str):
        shutil.rmtree(path)
