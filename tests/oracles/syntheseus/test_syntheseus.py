import pytest
import os
import shutil
import pickle
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

import sys
sys.path.append("../../../")

from oracles.dataclass import OracleComponentParameters
from oracles.synthesizability.syntheseus import Syntheseus

from utils.utils import set_seed_everywhere

# Expected Rxn-INSIGHT and NameRXN outputs for aripiprazole
# Rxn-INSIGHT: [
#       "Heteroatom Alkylation and Arylation, Williamson Ether Synthesis", 
#       "Heteroatom Alkylation and Arylation, N-alkylation of secondary amines with alkyl halides"
# ]
# NameRXN: [
#       "1.7.9 Williamson ether synthesis [O-substitution]", 
#       "1.6.2 Bromo N-alkylation [Heteroaryl N-alkylation]"
# ]

# Expected Rxn-INSIGHT and NameRXN outputs for amide molecule
# Rxn-INSIGHT: [
#       "Heteroatom Alkylation and Arylation, N-alkylation of secondary amines with alkyl halides", 
#       "Deprotection, Ester saponification (alkyl deprotection)",
#       "Acylation, Carboxylic acid with primary amine to amide"
# ]
# NameRXN: [
#       "1.6.4 Chloro N-alkylation [Heteroaryl N-alkylation]",
#       "6.2.1 CO2H-Et deprotection [RCO2H deprotections]",
#       "2.1.2 Carboxylic acid + amine condensation [N-acylation to amide]"
# ]

assert torch.cuda.is_available(), "CUDA is not available"
set_seed_everywhere(
    seed=0,
    device="cuda"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def aripiprazole_mol() -> Mol:
    return Chem.MolFromSmiles("O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1")

@pytest.fixture
def aripiprazole_hexane_chain_mol() -> Mol:
    """
    Arbitrarily added a hexane chain to aripiprazole as a *non-solvable* control molecule.
    """
    return Chem.MolFromSmiles("CCCCCCC1CC(=O)Nc2cc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)ccc21")

@pytest.fixture
def amide_mol() -> Mol:
    return Chem.MolFromSmiles("O=C(NC1CCCCC1)C1CCCN(Cc2ccccc2)C1")

@pytest.fixture
def route_file() -> str:
    return os.path.join(CURRENT_DIR, "aripiprazole-route.pkl")

@pytest.fixture
def base_oracle_params() -> dict:
    return {
        "name": "syntheseus",
        "weight": 1.0,
        "preliminary_check": False,
        "specific_parameters": {
            "syntheseus_env_name": "syntheseus-full",
            "reaction_model": "retroknn", 
            "building_blocks_file": os.path.join(CURRENT_DIR, "askcos-sigma-aldrich.smi"),
            "enforced_building_blocks": {
                "enforce_blocks": False,
                "enforced_building_blocks_file": os.path.join(CURRENT_DIR, "enforced-suzuki-stock-10.smi"),
                "enforce_start": False,
                "use_dense_reward": True,
                "reward_type": "tango_fms",
                "tango_weights": {
                    "tanimoto": 0.50,
                    "fg": 0.50,
                    "fms": 0.50
                }
            },
            "enforced_reactions": {
                "enforce_rxn_class_presence": False,
                "enforce_all_reactions": False,
                "rxn_insight_env_name": "rxn-insight",
                "use_namerxn": False,
                "namerxn_binary_path": "/home/jeff/saturn-dev/test/testing-namerxn/HazELNut/namerxn",
                "enforced_rxn_classes": [],
                "avoid_rxn_classes": [],
                "rxn_insight_extraction_script_path": "/home/jeff/saturn-dev/oracles/synthesizability/utils/extract_rxn_insight_info.py",
                "namerxn_extraction_script_path": "/home/jeff/saturn-dev/oracles/synthesizability/utils/extract_namerxn_info.py",
                "seed_reactions": False,
                "seed_reactions_file_folder": "/home/jeff/saturn-dev/test/mitsunobu-seeding-preprocessed"
            },
            "route_extraction_script_path": "/home/jeff/saturn-dev/oracles/synthesizability/utils/extract_syntheseus_route_data.py",
            "save_top_routes": False,
            "percentage_to_save": 0.005,
            "include_rxn_info": False,
            "time_limit_s": 180,
            "optimize_path_length": False,
            "parallelize": False,
            "max_workers": 4,
            "results_dir": os.path.join(CURRENT_DIR, "syntheseus_results")
        }
    }

def test_route_data(route_file):
    """
    Test the syntheseus-full environment can unpickle the route data correctly.
    """
    # Load the pickle file
    with open(route_file, "rb") as f:
        route = pickle.load(f)

    # Extract data from Syntheseus nodes
    syntheseus_route_data = {}
    for idx, node in enumerate(route):
        if hasattr(node, "mol"):
            syntheseus_route_data[idx] = {
                "smiles": node.mol.smiles,
                "depth": node.depth
            }
    
    # Check there are 5 molecule nodes
    assert len(syntheseus_route_data) == 5

def test_normal_synthesizability(aripiprazole_mol, aripiprazole_hexane_chain_mol, base_oracle_params):
    """
    Synthesizability with no constraints.
    """
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "binary"
    }
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(solved) == 1, "Expected 1 molecule in the SMILES list"
    assert (solved == np.array([2])).all(), "Expected the aripiprazole to be solved"

    # Test 2 molecules since Syntheseus results parsing is difference when > 1 input molecules
    solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, aripiprazole_hexane_chain_mol]),
        oracle_calls=1
    )
    assert len(solved) == 2, "Expected 2 molecules in the SMILES list"
    assert (solved == np.array([2, -99])).all(), "Expected the aripiprazole to be solved and the hexane chain to be unsolvable"

def test_rxn_class_synthesizability(aripiprazole_mol, amide_mol, base_oracle_params):
    """
    Synthesizability with a specific reaction class.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]

    # ----------------
    # Test Rxn-INSIGHT
    # ----------------
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = False
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 1
    assert (rxn_insight_solved == np.array([1])).all()

    # Test 2 molecules
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 2
    assert (rxn_insight_solved == np.array([1, 0])).all()

    # Test aripiprazole does not contain Amide reaction and that the Amide molecule does
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 2
    assert (rxn_insight_solved == np.array([0, 1])).all()
    
    # ------------
    # Test NameRXN
    # ------------
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1)
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([1])).all()

    # Test 2 molecules
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([1, 0])).all()

    # Test aripiprazole does not contain Amide reaction and that the Amide molecule does
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2, "Expected 2 molecules in the SMILES list"
    assert (namerxn_solved == np.array([0, 1])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C(NC1CCCCC1)C1CCCN(Cc2ccccc2)C1"
        ]
    }

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_all_rxn_class_synthesizability(aripiprazole_mol, amide_mol, base_oracle_params):
    """
    Synthesizability where *all* reactions in the synthesis graph match the user-specified reaction classes.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["alkylation"]

    # ----------------
    # Test Rxn-INSIGHT
    # ----------------
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = False
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 2
    assert (rxn_insight_solved == np.array([1, 0])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
        ]
    }

    # Negative control
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["suzuki"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 1
    assert (rxn_insight_solved == np.array([0])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }
    
    # ------------
    # Test NameRXN
    # ------------
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["alkylation"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([0])).all()  # NameRXN's label for the Williamson Ether Reaction does not reference "alkylation"

    # Positive control
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson", "alkylation"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1"
        ]
    }

    # Test that the matched SMILES tracker tracks both SMILES
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson", "alkylation", "deprotection", "to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([1, 1])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
            "O=C(NC1CCCCC1)C1CCCN(Cc2ccccc2)C1"
        ]
    }

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])


# avoid reaction
# avoid reaction while enforcing a reaction
# avoid reaction with enforcing all reactions

# enforce block
# enforce block + a reaction
# enforce block + all reactions
# enforce block + avoid reaction
# enforce block + a reaction + avoid reaction

# test the trackers as well