import numpy as np
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from oracles.xtb.geometry_optimizer import GeometryOptimizer


class HOMOLUMOGap(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.geometry_optimizer = GeometryOptimizer()

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        raw_homo_lumo_gap_values = []
        for mol in mols:
            try:
                added = False
                temp_dir, geometry_path, xtb_output = self.geometry_optimizer.optimize_geometry(mol)
                for line in xtb_output:
                    if "HOMO-LUMO GAP" in line:
                        raw_homo_lumo_gap_values.append(float(line.split()[-3]))
                        added = True
                        break
                
                if not added:
                    raw_homo_lumo_gap_values.append(99)

                # Delete temp file storing the geometry
                self.geometry_optimizer.clean_up_temp_dir(temp_dir)

            except Exception as e:
                # FIXME: Could be dangerous as 99 may actually be a good value
                raw_homo_lumo_gap_values.append(99)

        return np.array(raw_homo_lumo_gap_values, dtype=np.float32)
