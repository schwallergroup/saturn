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

# Expected Rxn-INSIGHT and NameRXN outputs for enforced block molecule
# Rxn-INSIGHT: [
#       "Reduction, Reduction of nitro groups to amines", 
#       "Oxidation, Oxidation or Dehydrogenation of Alcohols to Aldehydes and Ketones", 
#       "Deprotection, Hydrogenolysis of tertiary amines", 
#       "Acylation, N-alkylation of secondary amines with alkyl halides", 
#       "Acylation, Carboxylic acid with primary amine to amide"
# ]
# NameRXN: [
#       "7.1.1 Nitro to amino [Nitro to amine reduction]",
#       "8.1.5 Alcohol to ketone oxidation [Alcohols to aldehydes]",
#       "6.1.5 N-Bn deprotection [NH deprotections]",
#       "2.1.1 Amide Schotten-Baumann [N-acylation to amide]",  # Incorrect?
#       "2.1.2 Carboxylic acid + amine condensation [N-acylation to amide]"
# ]

assert torch.cuda.is_available(), "CUDA is not available"
set_seed_everywhere(
    seed=0,
    device="cuda"
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(CURRENT_DIR)
@pytest.fixture
def route_file() -> str:
    return os.path.join(CURRENT_DIR, "aripiprazole-route.pkl")

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
def enforced_block_mol() -> Mol:
    return Chem.MolFromSmiles("Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1")

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
                "enforced_building_blocks_file": os.path.join(CURRENT_DIR, "enforced-stock.smi"),
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
                # NOTE: The tests assume NameRXN version 3.7.3 (reaction classification may differ between different versions)
                # NOTE: Users should modify the path below to their local path
                "namerxn_binary_path": "<path to your namerxn binary executable>/HazELNut/namerxn",
                "enforced_rxn_classes": [],
                "avoid_rxn_classes": [],
                "rxn_insight_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_rxn_insight_info.py"),
                "namerxn_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_namerxn_info.py"),
                "seed_reactions": False,
                "seed_reactions_file_folder": os.path.join(CURRENT_DIR, "../../../test/mitsunobu-seeding-preprocessed")
            },
            "route_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_syntheseus_route_data.py"),
            "time_limit_s": 180,
            "minimize_path_length": False,
            "parallelize": False,
            "max_workers": 4,
            "results_dir": os.path.join(CURRENT_DIR, "syntheseus_results")
        }
    }

def test_route_data(route_file) -> None:
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

def test_normal_synthesizability(
    aripiprazole_mol, 
    aripiprazole_hexane_chain_mol, 
    base_oracle_params
) -> None:
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
    assert (solved == np.array([2])).all(), "Expected aripiprazole to be solved"

    # Test 2 molecules since Syntheseus results parsing is difference when > 1 input molecules
    solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, aripiprazole_hexane_chain_mol]),
        oracle_calls=1
    )
    assert len(solved) == 2, "Expected 2 molecules in the SMILES list"
    assert (solved == np.array([2, -99])).all(), "Expected aripiprazole to be solved and the hexane chain to be unsolvable"

def test_rxn_class_synthesizability(
    aripiprazole_mol, 
    amide_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability with a specific reaction class.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]

    # # ----------------
    # # Test Rxn-INSIGHT
    # # ----------------
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

def test_all_rxn_class_synthesizability(
    aripiprazole_mol, 
    amide_mol, 
    base_oracle_params
) -> None:
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

    # Sort in case set and list operators change the order
    assert sorted(syntheseus_oracle.matched_generated_smiles_with_rxn[1]) == sorted([
        "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
        "O=C(NC1CCCCC1)C1CCCN(Cc2ccccc2)C1"
    ])

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_avoid_rxn_class_synthesizability(
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability where *all* reactions in the synthesis graph match the user-specified reaction classes.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = False
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = ["suzuki"]

    # ---------------------------------------------
    # Test with Rxn-INSIGHT since it is open-source
    # ---------------------------------------------
    # The aripiprazole route does not contain a Suzuki reaction, so reward should be 1
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = False
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 1
    assert (rxn_insight_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
        ]
    }

    # Test jointly enforcing reaction class presence
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    rxn_insight_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(rxn_insight_solved) == 1
    assert (rxn_insight_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
        ]
    }

    # Test that the enforced reaction class is not in the route
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["wittig"]
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

    # Test enforcing all reactions
    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])
    
def test_enforced_block_synthesizability(
    enforced_block_mol, 
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability with enforced building block.
    
    NOTE: When testing dense reward matching, only TANGO is used.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = True
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforced_building_blocks_file"] = os.path.join(CURRENT_DIR, "enforced-stock.smi")
    # Test brute-force matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = False

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    # There are no reaction constraints so the matched_generated_smiles_with_rxn should be empty
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {}

    # Test dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    # There are no reaction constraints so the matched_generated_smiles_with_rxn should be empty
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {}

    # Test TANGO (0.5 Tanimoto and 0.5 FMS) dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["tango_weights"] = {
        "tanimoto": 0.50,
        "fg": 0.50,
        "fms": 0.50
    }
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert abs(block_solved[0] - 0.33370288) < 1e-6
    assert block_solved[1] == 1

    # Test TANGO (0.75 Tanimoto and 0.25 FMS) dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["tango_weights"] = {
        "tanimoto": 0.75,
        "fg": 0.25,
        "fms": 0.25
    }
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))

    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert abs(block_solved[0] - 0.22782705) < 1e-6
    assert block_solved[1] == 1

    # Test TANGO (1.0 Tanimoto and 0.0 FMS) dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["tango_weights"] = {
        "tanimoto": 1.0,
        "fg": 0.0,
        "fms": 0.0
    }
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))

    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert abs(block_solved[0] - 0.12195121) < 1e-6
    assert block_solved[1] == 1

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_enforced_block_rxn_class_synthesizability(
    enforced_block_mol, 
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability with enforced building block and reaction class.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = True
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforced_building_blocks_file"] = os.path.join(CURRENT_DIR, "enforced-stock.smi")
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert abs(block_solved[0] - 0.33370288) < 1e-6

    assert syntheseus_oracle.matched_generated_smiles == {
        1: []
    }
    # Even though aripirazole's route contains Williamson Ether Reaction, the enforced block reward is not 1.0, so the tracker below should not be populated
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Test enforced block molecule satisfies both conditions
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Test the scenario where the block is matched but the reaction is not
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["wittig"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([0])).all()  # The block is matched but the reaction is not, so the reward should be 0

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    # The reaction is not in the route, so this tracker should be empty
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Test the scenario where both block and reaction are matched but this time with NameRXN
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["amide schotten-baumann"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Test enforce start (expect False)
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = False
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_start"] = True
    # Test brute-force matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = False
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([0])).all()

    # Test dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert abs(block_solved[0] - 0.24447174) < 1e-6

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_enforced_block_all_rxn_class_synthesizability(
    enforced_block_mol, 
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability with enforced building block and all reaction classes.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = True
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforced_building_blocks_file"] = os.path.join(CURRENT_DIR, "enforced-stock.smi")
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson", "alkylation"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert abs(block_solved[0] - 0.33370288) < 1e-6

    assert syntheseus_oracle.matched_generated_smiles == {
        1: []
    }
    # Even though aripirazole's route contains Williamson Ether Reaction and Alkylation, the enforced block reward is not 1.0, so the tracker below should not be populated
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Negative control: When not enforcing *all* reaction classes, this should give a reward of 1.0
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([0])).all()  # Expect 0

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Positive control: Define a test to give reward of 1.0
    # Use Rxn-INSIGHT
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["reduction", "oxidation", "deprotection", "acylation"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    
    # Use NameRXN
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["reduction", "oxidation", "deprotection", "to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([1])).all()

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Test > 1 molecule 
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert (block_solved == np.array([0, 1])).all()  # Aripiprazole does not satisfy the reaction constraints
    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_enforced_block_avoid_rxn_class_synthesizability(
    enforced_block_mol, 
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Synthesizability with enforced building block and avoiding reaction classes.
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = True
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforced_building_blocks_file"] = os.path.join(CURRENT_DIR, "enforced-stock.smi")
    # Test brute-force matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = False
    # Use NameRXN
    base_oracle_params["specific_parameters"]["enforced_reactions"]["use_namerxn"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = ["stille"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert (block_solved == np.array([0, 1])).all()  # Aripiprazole does not contain the block

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Test dense reward matching
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert abs(block_solved[0] - 0.33370288) < 1e-6
    assert block_solved[1] == 1

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Test the scenario where a reaction is not successfully avoided
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = ["williamson","to amide"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert (block_solved == np.array([0, 0])).all()  # Now the enforced block molecule reward should be 0
    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Test jointly enforcing a reaction
    # Positive control
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["n-acylation to amide"]
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = ["mitsunobu"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert (block_solved == np.array([0, 1])).all()  # Expect only the enforced block molecule to have a reward of 1.0
    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }

    # Negative control
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["grignard"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    assert (block_solved == np.array([0, 0])).all()  # Expect both molecules to have a reward of 0
    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])

def test_path_length_minimization(
    enforced_block_mol, 
    aripiprazole_mol, 
    base_oracle_params
) -> None:
    """
    Path length minimization.
    Tests jointly enforcing blocks and reactions and avoiding reactions.
    """
    # General fixed parameters
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }
    base_oracle_params["specific_parameters"]["minimize_path_length"] = True

    # 1. Test enforced blocks
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = True
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforced_building_blocks_file"] = os.path.join(CURRENT_DIR, "enforced-stock.smi")
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["use_dense_reward"] = True

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert abs(block_solved[0] - 0.29584108) < 1e-6

    assert syntheseus_oracle.matched_generated_smiles == {
        1: []
    }

    # 2. Test jointly enforcing blocks and reactions
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["reduction", "oxidation", "deprotection", "acylation"]
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert abs(block_solved[0] - 0.39863019) < 1e-6

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    
    # 3. Test > 1 molecule with enforced reactions (no enforced blocks)
    #   * Positive control: aripiprazole (contains Williamson ether reaction)
    #   * Negative control: enforced block mol (does not contain Williamson ether reaction)
    base_oracle_params["specific_parameters"]["enforced_building_blocks"]["enforce_blocks"] = False
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = False
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 2
    # Enforced block mol does not contain the Williamson ether reaction, so the reward should be 0
    # Aripiprazole does contain the Williamson ether reaction, so the reward should be 1.0 * reward shaped path length
    assert abs(block_solved[0] - 0.88654037) < 1e-6
    assert block_solved[1] == 0

    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: [
            "O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1",
        ]
    }

    shutil.rmtree(base_oracle_params["specific_parameters"]["results_dir"])
