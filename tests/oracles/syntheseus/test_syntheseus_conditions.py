"""Separate tests only for reaction conditions (we only use NameRXN here)
"""
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

# Expected NameRXN and QUARC outputs for aripiprazole
# NameRXN: [
#       "1.7.9 Williamson ether synthesis [O-substitution]", 
#       "1.6.2 Bromo N-alkylation [Heteroaryl N-alkylation]"
# ]
# Conditions:
#  "agents": [
#     "O",
#     "CN(C)C=O",
#     "O=C(O[K])O[K]"
#  ],
#  "temperature": "[50.00, 60.00)"
#  "agents": [
#                 "O",
#                 "CN(C)C=O",
#                 "O=C(O[Na])O[Na]"
#             ],
#  "temperature": "[40.00, 50.00)",



# Expected NameRXN and QUARC outputs for amide molecule
# NameRXN: [
#       "1.6.4 Chloro N-alkylation [Heteroaryl N-alkylation]",
#       "6.2.1 CO2H-Et deprotection [RCO2H deprotections]",
#       "2.1.2 Carboxylic acid + amine condensation [N-acylation to amide]"
# ]
# "agents": [
#     "CN(C)C=O",
#     "O=C(O[K])O[K]"
# ],
# "temperature": "[100.00, 110.00)"
# "agents": [
#     "O",
#     "C1CCOC1",
#     "CO",
#     "O.[Li]O"
# ],
# "temperature": "[20.00, 30.00)"
# "agents": [
#     "O",
#     "CN(C)C=O",
#     "CCN(C(C)C)C(C)C",
#     "CN(C)C(On1nnc2cccnc21)=[N+](C)C.F[P-](F)(F)(F)(F)F"
# ],
# "temperature": "[20.00, 30.00)"


# Expected NameRXN and QUARC outputs for enforced block molecule
# NameRXN: [
#       "7.1.1 Nitro to amino [Nitro to amine reduction]",
#       "8.1.5 Alcohol to ketone oxidation [Alcohols to aldehydes]",
#       "6.1.5 N-Bn deprotection [NH deprotections]",
#       "2.1.1 Amide Schotten-Baumann [N-acylation to amide]",  # Incorrect?
#       "2.1.2 Carboxylic acid + amine condensation [N-acylation to amide]"
# # ]
#             "agents": [
#                 "CCOC(C)=O",
#                 "[Ni]"
#             ],
#             "temperature": "[20.00, 30.00)"
#             "agents": [
#                 "ClCCl",
#                 "O=[Mn]=O"
#             ],
#             "temperature": "[20.00, 30.00)"
#             "agents": [
#                 "CO",
#                 "O[Pd]O"
#             ],
#             "temperature": "[50.00, 60.00)"
#             "agents": [
#                 "ClCCl",
#                 "CCN(CC)CC"
#             ],
#             "temperature": "[20.00, 30.00)"
#             "agents": [
#                 "ClCCl",
#                 "CCN(CC)CC",
#                 "On1nnc2ccccc21",
#                 "CCN=C=NCCCN(C)C.Cl"
#             ],
#             "temperature": "[0.00, 10.00)"


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
                "use_namerxn": True,
                # NOTE: The tests assume NameRXN version 3.7.3 (reaction classification may differ between different versions)
                # NOTE: Users should modify the path below to their local path
                "namerxn_binary_path": "/bin/leadmine/HazELNut/namerxn",
                "enforced_rxn_classes": [],
                "avoid_rxn_classes": [],
                "rxn_insight_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_rxn_insight_info.py"),
                "namerxn_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_namerxn_info.py"),
                "seed_reactions": False,
                "seed_reactions_file_folder": os.path.join(CURRENT_DIR, "../../../test/mitsunobu-seeding-preprocessed"),
            },
            "enforced_conditions": {
                #NOTE: conditions block, not used here
                "avoid_conditions": [],
                "enforce_conditions": [],
                "enforce_temperature_range": [],
                "quarc_env_name": "quarc",
                "quarc_repo_path": "/home/sabanza/Documents/quarc",
                "condition_extraction_script_path": "../../../oracles/synthesizability/utils/extract_conditions.py"
            },
            "route_extraction_script_path": os.path.join(CURRENT_DIR, "../../../oracles/synthesizability/utils/extract_syntheseus_route_data.py"),
            "time_limit_s": 180,
            "minimize_path_length": False,
            "parallelize": False,
            "max_workers": 4,
            "results_dir": os.path.join(CURRENT_DIR, "syntheseus_results")
        }
    }

def test_conditions_enforcing(
        aripiprazole_mol,
        amide_mol,
        base_oracle_params
) -> None:
    """Test synthesizability with conditions only
    """
    # Parameters for reaction enforcing
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }

    # Test avoid DMF for (should be false)
    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = ["CN(C)C=O"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([0])).all()
    
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Test 2 molecules (both false)
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([0, 0])).all()
    
    # Test enforce condition and temperature range for 1 (should match)

    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = []
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_conditions"] = ["O"]
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = ["[40.00, 50.00)", "[50.00, 60.00)"]
    
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([1])).all()
    
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: ["O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1"]
    }

    # Test avoid and temperature range for 2
    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = ["CCN(C(C)C)C(C)C"]
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_conditions"] = []
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = ["[40.00, 50.00)", "[50.00, 60.00)"]
    
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([1, 0])).all()
    

def test_conditions_enforcing_and_rxn(
        aripiprazole_mol,
        amide_mol,
        base_oracle_params
) -> None:
    """Test synthesizability with conditions AND reaction enforcing"""

    # Test condition and match reaction for 2
    base_oracle_params["reward_shaping_function_parameters"] = {
        "transformation_function": "no_transformation"
    }

    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson"]
    
    # Avoid DMF (should be false)
    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = ["C1CCOC1"]
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = ["[40.00, 50.00)", "[50.00, 60.00)"]
    
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([1, 0])).all()
    
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: ["O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1"]
    }

    # Test condition and avoid reaction for 1
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = False
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = []
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = ["protection"]

    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 1
    assert (namerxn_solved == np.array([1])).all()
    

    # Test condition and all reactions for 2

    # Enforce all williamson and alkylation
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_rxn_class_presence"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["avoid_rxn_classes"] = []
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["williamson", "alkylation"]

    # Enforce presence of water and avoid benzene
    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = ["c1ccccc1", "Cc1ccccc1"]
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_conditions"] = ["O"]
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = []
    
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    namerxn_solved = syntheseus_oracle(
        mols=np.array([aripiprazole_mol, amide_mol]),
        oracle_calls=1
    )
    assert len(namerxn_solved) == 2
    assert (namerxn_solved == np.array([1, 0])).all()
    
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: ["O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1"]
    }

def test_conditions_enforcing_and_rxn_and_block(
        enforced_block_mol,
        aripiprazole_mol,
        base_oracle_params
) -> None:
    """Tests conditions, reaction enforcing and block enforcing
    """
    # Test conditions, rxn and block for one
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = ["[40.00, 50.00)", "[50.00, 60.00)"]
    
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
    # Even though aripirazole's route contains Williamson Ether Reaction and is in temperature range, the enforced block reward is not 1.0, so the tracker below should not be populated
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }

    # Test enforced block molecule satisfies all conditions
    base_oracle_params["specific_parameters"]["enforced_conditions"]["enforce_temperature_range"] = ["[0.00, 10.00)", "[20.00, 30.00)","[50.00, 60.00)"]
    
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
    assert (block_solved == np.array([0])).all()  # The block and condition are matched but the reaction is not, so the reward should be 0

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    # The reaction is not in the route, so this tracker should be empty
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }
    
    # Test the scenario where the block and reaction is matched but the condition is not
    base_oracle_params["specific_parameters"]["enforced_conditions"]["avoid_conditions"] = ["O=[Mn]=O", 
                                                                                           "[Ni]"]
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforce_all_reactions"] = True
    base_oracle_params["specific_parameters"]["enforced_reactions"]["enforced_rxn_classes"] = ["to amide",
                                                                                               "alcohol to ketone oxidation",
                                                                                               "deprotection",
                                                                                               "reduction"]
    
    syntheseus_oracle = Syntheseus(OracleComponentParameters(**base_oracle_params))
    block_solved = syntheseus_oracle(
        mols=np.array([enforced_block_mol]),
        oracle_calls=1
    )
    assert len(block_solved) == 1
    assert (block_solved == np.array([0])).all()  # The block and rxn are matched but the condition is not, so the reward should be 0

    assert syntheseus_oracle.matched_generated_smiles == {
        1: [
            "Cc1ccc(CC(=O)Nc2ccc(C(=O)N3CCC(=O)C3)o2)cc1",
        ]
    }
    # The reaction is not in the route, so this tracker should be empty
    assert syntheseus_oracle.matched_generated_smiles_with_rxn == {
        1: []
    }