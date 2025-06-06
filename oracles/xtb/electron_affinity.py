import numpy as np
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from oracles.xtb.geometry_optimizer import GeometryOptimizer
from morfeus import read_xyz, XTB

class ElectronAffinity(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.geometry_optimizer = GeometryOptimizer()

    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        raw_ea_values = []
        for mol in mols:
            try:
                temp_dir, geometry_path, xtb_output = self.geometry_optimizer.optimize_geometry(mol)
                elements, coordinates = read_xyz(geometry_path)
                xtb = XTB(elements, coordinates)
                # delete temp file storing the geometry
                self.geometry_optimizer.clean_up_temp_dir(temp_dir)
                raw_ea_values.append(xtb.get_ea(corrected=True))

            except Exception:
                # FIXME: could be dangerous as 0.0 may actually be a good value
                raw_ea_values.append(0.0)

        return np.array(raw_ea_values, dtype=np.float32)
