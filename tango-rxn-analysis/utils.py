"""Utils for analysis"""
from typing import Dict, List, Tuple, Union
import os
import logging
import subprocess
import json
import ast
import random
import datetime
import pandas as pd
import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import DataStructs

import matplotlib.pyplot as plt
import more_itertools as mit
from matplotlib import pyplot as plt
from collections import defaultdict

import warnings
from rdkit import RDLogger
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
RDLogger.DisableLog('rdApp.*')

MORGAN_RADIUS = 2
MORGAN_BITS = 1024


# -------------------------
# General Utility Functions
# -------------------------
def setup_logging(logging_path: str):
    """Sets up logging to a file and console."""
    logging.basicConfig(filename=logging_path, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger("").addHandler(console)

def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize the given SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return smiles

def df_remove_duplicate_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate SMILES from the given DataFrame.
    """
    df["canonical_smiles"] = df["smiles"].apply(lambda x: canonicalize_smiles(x) if x is not None and x != "" else None)
    df = df.drop_duplicates(subset=["canonical_smiles"])
    return df

def get_morgan_fingerprints(mols: List[Mol], as_list: bool = False) -> Union[list, np.array]:
    """
    Get the Morgan fingerprints for the given molecules.
    """
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, MORGAN_RADIUS, MORGAN_BITS) for m in mols]
    if as_list:
        return fps
    else:
        return np.array(fps)

def get_wall_time(lines: List[str]) -> float:
    total_time = float(lines[-1].split()[-2])
    # Extract timestamps from the last two lines
    oracle_calls_time = datetime.datetime.strptime(
        lines[-2].split()[0] + " " + lines[-2].split()[1].split(",")[0], 
        "%Y-%m-%d %H:%M:%S"
    )
    top_graphs_time = datetime.datetime.strptime(
        lines[-1].split()[0] + " " + lines[-1].split()[1].split(",")[0],
        "%Y-%m-%d %H:%M:%S"
    )
    time_diff = (top_graphs_time - oracle_calls_time).total_seconds()
    # Subtract time for top graphs extraction from generation time (if applicable)
    if time_diff > 300:  # Top graphs should take at least 5 minutes
        return total_time - time_diff
    else:
        return total_time

def ligand_efficiency(
    mols: List[Mol], 
    docking_scores: List[float]
) -> List[float]:
    """
    Calculate the ligand efficiencies for the molecules.
    """
    ligand_efficiencies = []
    for idx, mol in enumerate(mols):
        try:
            ligand_efficiencies.append(abs(docking_scores[idx] / mol.GetNumHeavyAtoms()))
        except Exception:
            continue

    assert len(ligand_efficiencies) == len(mols), "Mismatch in ligand efficiencies and number of molecules."

    return ligand_efficiencies

# ---------------------------
# Diversity Utility Functions
# ---------------------------
def num_unique_murcko_scaffolds(mols: List[Mol]) -> int:
    """
    Count the number of unique Murcko scaffolds in the input SMILES list.
    """
    scaffolds = set()
    for mol in mols:
        try:
            scaffold = GetScaffoldForMol(mol)
            scaffolds.add(Chem.MolToSmiles(scaffold, canonical=True))
        except Exception:
            continue

    return len(scaffolds)

# From https://github.com/molecularsets/moses/blob/master/moses/metrics/metrics.py
def average_agg_tanimoto(
    stock_vecs: np.array, 
    gen_vecs: np.array,
    batch_size: int = 5000, 
    agg: str = "max",
    device: str = "cpu", 
    p: int = 1
) -> float:
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ["max", "mean"], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == "max":
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == "mean":
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == "mean":
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

# From https://github.com/molecularsets/moses/blob/master/moses/metrics/metrics.py
def internal_diversity(
    mols: List[Mol], 
    n_jobs: int = 1, 
    device: str = "cpu", 
    fp_type: str = "morgan",
    gen_fps: np.array = None, 
    p: int = 1
) -> float:
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    fps = get_morgan_fingerprints(mols)
    return 1 - (average_agg_tanimoto(fps, fps, agg="mean", device=device, p=p)).mean()
        
def get_ncircle(df):
    if "FPS" not in df:
        df["FPS"] = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df["MOL"]]
    return NCircles().measure(df["FPS"])

def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)

class NCircles():
    def __init__(self, threshold=0.75):
        super().__init__()
        self.sim_mat_func = similarity_matrix_tanimoto
        self.t = threshold
    
    def get_circles(self, args):
        vecs, sim_mat_func, t = args
        
        circs = []
        for vec in vecs:
            if len(circs) > 0:
                dists = 1. - sim_mat_func([vec], circs)
                if dists.min() <= t: continue
            circs.append(vec)
        return circs

    def measure(self, vecs, n_chunk=64):
        for i in range(3):
            vecs_list = [list(c) for c in mit.divide(n_chunk // (2 ** i), vecs)]
            args = zip(vecs_list, 
                       [self.sim_mat_func] * len(vecs_list), 
                       [self.t] * len(vecs_list))
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs)

def write_out_top_syntheseus_graphs(
    oracle_history: pd.DataFrame,
    syntheseus_folder: str,
    syntheseus_path_script: str,
    smiles_rxn_tracker: dict
) -> None:
    """
    Extract the Syntheseus synthesis graph information for the highest reward molecules (given they satisfy all reaction constraints).
    The purpose of this function is to automatically allow the user to visualize the synthesis routes for the top molecules.
    """
    # Output JSON with all relevant metrics and information
    output = {}

    syntheseus_outputs = os.listdir(syntheseus_folder)

    for idx, (_, row) in enumerate(oracle_history.iterrows()):
        oracle_calls = int(row["oracle_calls"])
        generated_smiles = row["smiles"]
        # Extract the Oracle raw values
        reward = {
            "reward": row["reward"],
            **{col: row[col] for col in row.index if "raw_values" in col}
        }
        
        # NOTE: This code is used to find the exact path to the syntheseus output for the generated molecule
        # Find the Syntheseus output folder with the closest *smaller* number of oracle calls
        # FIXME: This is because currently, oracle calls is incremented before the Oracle History is updated. Fix this in the future. This is inelegant
        closest_smaller_oracle_calls_folder = max(
            [
                folder for folder in syntheseus_outputs 
                if not folder.endswith(".json") and 
                not folder.endswith(".pdf") and 
                not "graphs" in folder and
                int(folder.split("_")[-1]) < oracle_calls
            ],
            key=lambda x: int(x.split("_")[-1])
        )

        # All the Mols matching the oracle calls
        mol_folder = os.listdir(os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder))
        # Loop through each to find the correct molecule
        for individual_mol_folder in mol_folder:
            added = False
            all_output_files = os.listdir(os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder))
            # Check if the Mol is solved
            if "route_0.pkl" in all_output_files:
                # Extract the route Mol data
                route = extract_syntheseus_route_data(
                    route_path=os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
                    data_type="mol",
                    syntheseus_script_path=syntheseus_path_script,
                )
                for node, node_data in route.items():
                    # Check if the generated SMILES is in the route
                    if node_data["depth"] == 0 and canonicalize_smiles(node_data["smiles"]) == canonicalize_smiles(generated_smiles):
                        added = True  
                        if added:
                            break
                if added:
                    break

        # Get Synthesis data
        synthesis_data = extract_syntheseus_route_data(
            route_path=os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
            data_type="all",
            syntheseus_script_path=syntheseus_path_script
        )     

        # For each Reaction node (is_rxn = True), extract the reaction information if the user specified to include reaction information
        for node, node_data in synthesis_data.items():
            # Nodes are traversed from the root because the extraction script above sorts them by depth so the first one should alwayas be the generated molecule
            # FIXME: Canonicalization calls below should be redundant
            if node_data["depth"] == 0:
                generated_smiles = canonicalize_smiles(node_data["mol_smiles"])
            elif node_data["is_rxn"]:
                # Extract the enforced block, rxn_class, and rxn_name based on the matching rxn_smiles
                rxn_info = smiles_rxn_tracker[generated_smiles]
                specific_enforced_block = rxn_info["enforced_block"]

                for depth, individual_rxn_info in rxn_info.items():
                    if individual_rxn_info["rxn_smiles"] == node_data["rxn_smiles"]:
                        node_data["rxn_class"] = individual_rxn_info["rxn_class"]
                        node_data["rxn_name"] = individual_rxn_info["rxn_name"]
                        break

        # Construct the JSON for the current molecule
        output[generated_smiles] = {
            "reward": reward,
            "synthesis_data": synthesis_data,
            "enforced_block": specific_enforced_block
        }
    return output

def extract_syntheseus_route_data(
    route_path: str,
    data_type: str,
    syntheseus_script_path: str
) -> Dict[str, Union[str, int]]:
    
    # Read Syntheseus route data from the pickle file and extract the Mols or Reactions
    # HACK: This (temporary) solution enables reading the pickled data *without* installing Syntheseus into the Saturn environment
    extraction_result = subprocess.run([
        "conda", 
        "run", 
        "-n",
        "syntheseus-full", 
        "python", 
        syntheseus_script_path, 
        # NOTE: We set Syntheseus to terminate after 1 route is found, so the index is always 0
        route_path,
        data_type,  # "mol", "rxn", or "all"
    ], capture_output=True, text=True)

    # Check for errors
    assert extraction_result.returncode == 0, f"Error during Syntheseus route ({data_type}) data extraction: {extraction_result.stderr}"
    route = json.loads(extraction_result.stdout)

    return route

def get_run_data(path: str) -> Tuple[bool, bool, str]:
    """
    Take run .json file and get info related to reaction and building blocks.
    """
    files = os.listdir(path)
    
    # Get JSON with file
    file = [file for file in files if file.endswith(".json") and "run" in file][0]

    with open(os.path.join(path, file), "r") as f:
        data = json.load(f)
    
    # Get Syntheseus info
    syntheseus_info = [component for component in data["oracle"]["components"] if component["name"] == "syntheseus"][0]["specific_parameters"]
    
    enforce_reactions = syntheseus_info["enforced_reactions"]["enforce_rxn_class_presence"]
    enforce_building_blocks = syntheseus_info["enforced_building_blocks"]["enforce_blocks"]
    enforced_building_blocks_file = syntheseus_info["enforced_building_blocks"]["enforced_building_blocks_file"]
    
    return enforce_reactions, enforce_building_blocks, enforced_building_blocks_file

def plot_rxn_evolution(
    smiles_rxn_tracker: Dict[str, Dict[str, str]],
    enforced_rxn: str,
    save_dir: str,
    experiment_name: str
) -> None:
    """Plot evolution of reaction classes."""
    # TODO: subplot for each seed
    # Load data
    seed = 0
    experiment_path = f"test_files/{enforced_rxn}/seed{seed}"
    oracle_history = pd.read_csv(f"{experiment_path}/oracle_history.csv")
    oracle_history["canonical_smiles"] = oracle_history["smiles"].apply(canonicalize_smiles)

    # Track reactions by class over time
    stats = defaultdict(list)
    rxn_smiles = defaultdict(list)

    for smiles, rxn_info in smiles_rxn_tracker.items():
        oracle_calls = rxn_info["oracle_calls"]
        for depth, info in rxn_info.items():
            rxn_class = info["rxn_class"]
            rxn_name = info["rxn_name"]
                
            if oracle_calls is not None and rxn_class != "Unrecognized":
                stats[(rxn_class, rxn_name)].append(oracle_calls)
                rxn_smiles[(rxn_class, rxn_name)].append(info["rxn_smiles"])

    # Sort each reaction class by oracle calls
    for rxn_class_name in stats:
        stats[rxn_class_name] = sorted(stats[rxn_class_name])

    # Sort reaction classes by count in descending order
    sorted_stats = sorted(stats.items(), key=lambda x: len(x[1]), reverse=True)

    # Filter for reactions with count > 500 and take top 10
    sorted_stats = [(k,v) for k,v in sorted_stats if len(v) > 500][:10]

    colours = [
        "#2ecc71", "#3498db", "#9b59b6", "#f1c40f", "#e67e22", 
        "#1abc9c", "#34495e", "#95a5a6", "#d35400", "#c0392b"
    ]
    enforced_colour = "#e74c3c"  # Bright red for enforced reaction

    # Plot cumulative reactions over time
    plt.figure(figsize=(16,8))

    # Add legend entry explaining format
    plt.plot([], [], " ", label="<Reaction Class>\n<Reaction Name>")

    # Plot filtered reaction classes
    for idx, ((rxn_class, rxn_name), calls) in enumerate(sorted_stats):
        if enforced_rxn.lower() in rxn_class.lower() or enforced_rxn.lower() in rxn_name.lower():
            plt.plot(calls, range(1, len(calls)+1), 
                    label=f"{rxn_class}\n{rxn_name} (n={len(calls)})", 
                    color=enforced_colour, alpha=1.0, linewidth=3)
        else:
            plt.plot(calls, range(1, len(calls)+1), 
                    label=f"{rxn_class}\n{rxn_name} (n={len(calls)})", 
                    color=colours[idx], alpha=1.0, linewidth=1)

    plt.xlabel("Number of Oracle Calls", fontsize=12, fontweight="bold")
    plt.ylabel("Cumulative Number of Reactions", fontsize=12, fontweight="bold")
    plt.title("Cumulative Growth of Different Reaction Types", fontsize=14, fontweight="bold")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=12)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, f"{experiment_name}-rxn-evolution.png"))

def count_rxn_graph(top_graphs: Dict[str, Union[str, float]]) -> Union[Dict[str, int], List[int]]:
    """
    Count number of reaction classes and number of reaction steps for each top graph.
    """
    rxn_count = dict()
    rxn_steps = []

    for key, value in top_graphs.items():

        synthesis_graph = top_graphs[key]["synthesis_data"]
        steps = 0

        for node, value in synthesis_graph.items():
            if value["is_rxn"]:
                rxn_name = value["rxn_name"]
                if rxn_name not in rxn_count:
                    rxn_count[rxn_name] = 1
                else:
                    rxn_count[rxn_name] += 1
                    
                steps += 1

        rxn_steps.append(steps)
    
    return rxn_count, rxn_steps

def plot_rxn_classes(
    rxn_count: Dict[str, int],
    save_dir: str,
    experiment_name: str
) -> None:
    """Plot reaction class distribution and save barplot."""
    # Save the raw counts
    with open(os.path.join(save_dir, f"{experiment_name}-rxn-distribution.json"), "w") as f:
        json.dump(rxn_count, f, indent=4)

    # Extract categories and counts
    categories, counts = list(rxn_count.keys()), list(rxn_count.values())

    # Compute proportions
    total = sum(counts)
    proportions = [count / total for count in counts]

    # Create the barplot
    plt.figure(figsize=(8, 8))
    bars = plt.bar(categories, proportions, width=0.5, edgecolor="black")

    # Annotate the bars with absolute counts
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                    f"{count}", ha="center", va="bottom")

    # Set axis labels and title
    plt.xlabel("Reaction Classes", fontsize=12, fontweight="bold")
    plt.ylabel("Proportion", fontsize=12, fontweight="bold")
    plt.title("Proportion of Reaction Classes", fontsize=14, fontweight="bold")

    # Adjust layout and display the plot
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(os.path.join(save_dir, f"{experiment_name}-rxn-distribution.png"))

# NOTE: Legacy function: not used anymore
def legacy(
    oracle_history: pd.DataFrame,
    syntheseus_folder: str,
    enforce_building_blocks: bool,
    enforced_building_blocks_file: str,
    syntheseus_path_script: str,
    rxn_insight_path_script: str,
    use_namerxn: bool,
    namerxn_binary_path: str,
    name_rxn_path_script: str
) -> None:
    """
    Extract the Syntheseus synthesis graph PDF files for the highest reward molecules (given they satisfy all reaction constraints).
    The purpose of this function is to automatically allow the user to visualize the synthesis routes for the top molecules.
    """
    # Output JSON with all relevant metrics and information
    output = {}
    
    # Loop through each top generated SMILES, extract the Syntheseus graph, and track which enforced smiles is visited (if applicable)
    if enforce_building_blocks:
        enforced_building_blocks_smiles = set([canonicalize_smiles(s) for s in open(enforced_building_blocks_file).readlines()])

    syntheseus_outputs = os.listdir(syntheseus_folder)

    for idx, (_, row) in enumerate(oracle_history.iterrows()):
        oracle_calls = int(row["oracle_calls"])
        generated_smiles = row["smiles"]
        # Extract the Oracle raw values
        reward = {
            "reward": row["reward"],
            **{col: row[col] for col in row.index if "raw_values" in col}
        }
        
        # Find the Syntheseus output folder with the closest *smaller* number of oracle calls
        # FIXME: This is because currently, oracle calls is incremented before the Oracle History is updated. Fix this in the future. This is inelegant
        closest_smaller_oracle_calls_folder = max(
            [
                folder for folder in syntheseus_outputs 
                if not folder.endswith(".json") and 
                not folder.endswith(".pdf") and 
                not "graphs" in folder and
                int(folder.split("_")[-1]) < oracle_calls
            ],
            key=lambda x: int(x.split("_")[-1])
        )

        # All the Mols matching the oracle calls
        mol_folder = os.listdir(os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder))
        # Loop through each to find the correct molecule
        for individual_mol_folder in mol_folder:
            added = False
            all_output_files = os.listdir(os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder))
            # Check if the Mol is solved
            if "route_0.pkl" in all_output_files:
                # Extract the route Mol data
                route = extract_syntheseus_route_data(
                    route_path=os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
                    data_type="mol",
                    syntheseus_script_path=syntheseus_path_script,
                )
                # Track which enforced block is visited (if applicable)
                specific_enforced_block = None

                for node, node_data in route.items():
                    # Check if the generated SMILES is in the route
                    if node_data["depth"] == 0 and canonicalize_smiles(node_data["smiles"]) == canonicalize_smiles(generated_smiles):
                        added = True  
                        if enforce_building_blocks:
                            # Extract which enforced block is present
                            for intermediate_node, intermediate_node_data in route.items():
                                canonical_intermediate_smiles = canonicalize_smiles(intermediate_node_data["smiles"])  # Canonicalize in case
                                if canonical_intermediate_smiles in enforced_building_blocks_smiles:
                                    specific_enforced_block = canonical_intermediate_smiles
                                    break
                        if added:
                            break
                if added:
                    break

        # Get Synthesis data
        synthesis_data = extract_syntheseus_route_data(
            route_path=os.path.join(syntheseus_folder, closest_smaller_oracle_calls_folder, individual_mol_folder, "route_0.pkl"),
            data_type="all",
            syntheseus_script_path=syntheseus_path_script
        )     

        # For each Reaction node (is_rxn = True), extract the reaction information if the user specified to include reaction information
        for node, node_data in synthesis_data.items():
            if node_data["is_rxn"]:
                extraction_result = subprocess.run([
                    "conda",
                    "run", 
                    "-n",
                    "rxn-insight", 
                    "python", 
                    rxn_insight_path_script, 
                    # Pass the rxn SMILES extracted from the Syntheseus route
                    node_data["rxn_smiles"]
                ], capture_output=True, text=True)

                # Check for errors
                assert extraction_result.returncode == 0, f"Error during Rxn-INSIGHT reaction information extraction: {extraction_result.stderr}"
                rxn_info = ast.literal_eval(extraction_result.stdout)
                node_data["rxn_class"] = rxn_info["CLASS"]
                node_data["rxn_name"] = rxn_info["NAME"]

        # Construct the JSON for the current molecule
        output[generated_smiles] = {
            "reward": reward,
            "synthesis_data": synthesis_data,
            "enforced_block": specific_enforced_block
        }
    return output