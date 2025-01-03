import numpy as np
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from oracles.xtb.geometry_optimizer import GeometryOptimizer


class MolecularDipole(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.geometry_optimizer = GeometryOptimizer()

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        raw_molecular_dipole_values = []
        for mol in mols:
            try:
                temp_dir, geometry_path, xtb_output = self.geometry_optimizer.optimize_geometry(mol)
                for idx, line in enumerate(xtb_output):
                    if "molecular dipole" in line:
                        dipole_line = xtb_output[idx+3]
                        raw_molecular_dipole_values.append(float(dipole_line.split()[-1]))
                        break

                # Delete temp file storing the geometry
                self.geometry_optimizer.clean_up_temp_dir(temp_dir)

            except Exception as e:
                # FIXME: Could be dangerous as 0.0 may actually be a good value
                raw_molecular_dipole_values.append(0.0)

        return np.array(raw_molecular_dipole_values, dtype=np.float32)
