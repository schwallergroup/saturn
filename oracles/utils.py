"""
Based on implementation from https://github.com/MolecularAI/reinvent-scoring/blob/main/reinvent_scoring/scoring/score_components/score_component_factory.py.
Contains utility function to initialize OracleComponents.
"""
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters

# Similarity metrics
from oracles.similarity.tanimoto_similarity import TanimotoSimilarity
from oracles.similarity.jaccard_distance import JaccardDistance
from oracles.similarity.jaccard_distance_dataset import JaccardDistanceDataset

# Physchem properties
from oracles.physchem.aliphatic_rings import NumAliphaticRings
from oracles.physchem.aromatic_rings import NumAromaticRings
from oracles.physchem.hba import NumHydrogenBondAcceptors
from oracles.physchem.hbd import NumHydrogenBondDonors
from oracles.physchem.mw import MolecularWeight
from oracles.physchem.qed import QED
from oracles.physchem.rings import NumRings
from oracles.physchem.rotatable_bonds import NumRotatableBonds
from oracles.physchem.stereocenters import NumStereocenters
from oracles.physchem.tpsa import tPSA

# Structural
from oracles.structural.matching_substructure import MatchingSubstructure
from oracles.structural.smarts_alerts import SMARTSAlert

# Synthesizability
from oracles.synthesizability.sa_score import SAScore
from oracles.synthesizability.aizynthfinder import AiZynthFinder

# Docking
from oracles.docking.dockstream import DockStream
from oracles.docking.quickvina2_gpu import QuickVina2_GPU

# xTB electronic properties
from oracles.xtb.chemical_potential import ChemicalPotential
from oracles.xtb.dipole import Dipole
from oracles.xtb.electron_affinity import ElectronAffinity
from oracles.xtb.electrophilicity_index import ElectrophilicityIndex
from oracles.xtb.electrophilicity import Electrophilicity
from oracles.xtb.hardness import Hardness
from oracles.xtb.homo import HOMO
from oracles.xtb.ionization_potential import IonizationPotential
from oracles.xtb.lumo import LUMO
from oracles.xtb.nucleophilicity_index import NucleophilicityIndex
from oracles.xtb.nucleophilicity import Nucleophilicity

# GEAM's oracle
from oracles.docking.geam_oracle import GEAMOracle
from oracles.docking.quickvina2 import QuickVina2


def construct_oracle_component(oracle_component_parameters: OracleComponentParameters) -> OracleComponent:
    """
    Matches the OracleComponent name and returns the OracleComponent class.
    """
    name = oracle_component_parameters.name
    # Similarity metrics
    if name == "tanimoto_similarity":
        return TanimotoSimilarity(oracle_component_parameters)
    elif name == "jaccard_distance":
        return JaccardDistance(oracle_component_parameters)
    elif name == "jaccard_distance_dataset":
        return JaccardDistanceDataset(oracle_component_parameters)
    # Physchem properties
    elif name == "num_aliphatic_rings":
        return NumAliphaticRings(oracle_component_parameters)
    elif name == "num_aromatic_rings":
        return NumAromaticRings(oracle_component_parameters)
    elif name == "num_hba":
        return NumHydrogenBondAcceptors(oracle_component_parameters)
    elif name == "num_hbd":
        return NumHydrogenBondDonors(oracle_component_parameters)
    elif name in ["mw", "molecular_weight"]:
        return MolecularWeight(oracle_component_parameters)
    elif name == "qed":
        return QED(oracle_component_parameters)
    elif name == "num_rings":
        return NumRings(oracle_component_parameters)
    elif name == "num_rotatable_bonds":
        return NumRotatableBonds(oracle_component_parameters)
    elif name == "num_stereocenteres":
        return NumStereocenters(oracle_component_parameters)
    elif name == "tpsa":
        return tPSA(oracle_component_parameters)
    # Structural
    elif name == "matching_substructure":
        return MatchingSubstructure(oracle_component_parameters)
    elif name == "smarts_alerts":
        return SMARTSAlert(oracle_component_parameters)
    # Synthesizability
    elif name == "sa_score":
        return SAScore(oracle_component_parameters)
    elif name == "aizynthfinder":
        return AiZynthFinder(oracle_component_parameters)
    # Docking
    elif name == "dockstream":
        return DockStream(oracle_component_parameters)
    elif name == "quickvina2_gpu":
        return QuickVina2_GPU(oracle_component_parameters)
    # xTB electronic properties
    elif name == "chemical_potential":
        return ChemicalPotential(oracle_component_parameters)
    elif name == "dipole":
        return Dipole(oracle_component_parameters)
    elif name == "electron_affinity":
        return ElectronAffinity(oracle_component_parameters)
    elif name == "electrophilicity_index":
        return ElectrophilicityIndex(oracle_component_parameters)
    elif name == "electrophilicity":
        return Electrophilicity(oracle_component_parameters)
    elif name == "hardness":
        return Hardness(oracle_component_parameters)
    elif name == "homo":
        return HOMO(oracle_component_parameters)
    elif name == "ionization_potential":
        return IonizationPotential(oracle_component_parameters)
    elif name == "lumo":
        return LUMO(oracle_component_parameters) 
    elif name == "nucleophilicity_index":
        return NucleophilicityIndex(oracle_component_parameters)
    elif name == "nucleophilicity":
        return Nucleophilicity(oracle_component_parameters)
    # GEAM's oracle
    elif name == "geam":
        return GEAMOracle(oracle_component_parameters)
    elif name == "quickvina2":
        return QuickVina2(oracle_component_parameters)
    # TODO: Pharmacophore and Shape Match
    # TODO: MD
    else:
        raise NotImplementedError(f"Oracle: {name} is not implemented.")
