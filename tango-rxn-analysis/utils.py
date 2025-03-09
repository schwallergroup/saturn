"""Utils for analysis"""
from typing import Dict, List, Tuple, Union
import os
import shutil
import subprocess
import logging
import json
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
    top_oracle_history: pd.DataFrame,
    smiles_rxn_tracker: dict
) -> Dict[str, Dict[str, Union[str, int]]]:
    """
    Extract the Syntheseus synthesis graph information for the highest reward molecules (given they satisfy all reaction constraints).
    The purpose of this function is to automatically allow the user to visualize the synthesis routes for the top molecules.
    """
    # Output JSON with all relevant metrics and information
    output = {}

    for idx, (_, row) in enumerate(top_oracle_history.iterrows()):
        generated_smiles = canonicalize_smiles(row["smiles"])
        # Extract the Oracle raw reward values
        reward = {
            "reward": row["reward"],
            **{col: row[col] for col in row.index if "raw_values" in col}
        }

        # Extract the synthesis constraints information (enforced block and/or reaction)
        synthesis_data = smiles_rxn_tracker[generated_smiles]

        # Construct the JSON for the current molecule
        output[generated_smiles] = {
            "reward": reward,
            "enforced_block": synthesis_data["enforced_block"],
            # FIXME: This is due to the current GUI expecting a certain data structure
            "synthesis_data": {k:v for k,v in synthesis_data.items() if k != "enforced_block"},
        }

    return output

def get_run_data(path: str) -> Tuple[bool, List[str], bool, str]:
    """
    Take run .json file and get info related to reaction and building blocks.
    """
    files = os.listdir(path)
    
    # Get JSON with file
    # TODO: Make the run configuration string matching more robust
    file = [file for file in files if file.endswith(".json") and "run" in file][0]

    with open(os.path.join(path, file), "r") as f:
        data = json.load(f)
    
    # Get Syntheseus info
    syntheseus_info = [component for component in data["oracle"]["components"] if component["name"] == "syntheseus"][0]["specific_parameters"]
    
    enforce_reactions = syntheseus_info["enforced_reactions"]["enforce_rxn_class_presence"]
    enforced_reaction_classes = syntheseus_info["enforced_reactions"]["enforced_rxn_classes"]
    enforce_building_blocks = syntheseus_info["enforced_building_blocks"]["enforce_blocks"]
    enforced_building_blocks_file = syntheseus_info["enforced_building_blocks"]["enforced_building_blocks_file"]
    
    return enforce_reactions, enforced_reaction_classes, enforce_building_blocks, enforced_building_blocks_file

def plot_rxn_evolution(
    seeds_paths: List[str],
    enforced_rxn: str,
    save_dir: str,
    experiment_name: str
) -> None:
    """Plot evolution of reaction classes."""
    # Load data
    for seed_path in seeds_paths:
        if not os.path.exists(os.path.join(seed_path, "oracle_history.csv")):
            continue
        
        oracle_history = pd.read_csv(f"{seed_path}/oracle_history.csv")
        oracle_history["canonical_smiles"] = oracle_history["smiles"].apply(canonicalize_smiles)

        # Track reactions by class over time
        smiles_rxn_tracker = json.load(open(os.path.join(seed_path, "syntheseus_results", "smiles_rxn_tracker.json"), "r"))
        rxn_stats = defaultdict(list)
        all_rxn_steps = []

        for smiles, rxn_info in smiles_rxn_tracker.items():
            for attribute, attribute_value in rxn_info.items():
                if attribute == "oracle_calls":
                    oracle_calls = attribute_value
                elif attribute == "rxn_steps":
                    rxn_steps = attribute_value
                    all_rxn_steps.append(rxn_steps)
                elif isinstance(attribute_value, dict):
                    if attribute_value["is_rxn"]:
                        rxn_class = attribute_value["rxn_class"]
                        rxn_name = attribute_value["rxn_name"]
                        if oracle_calls is not None and rxn_class != "Unrecognized":
                            rxn_stats[(rxn_class, rxn_name)].append(oracle_calls)
                    
        logging.info(f"{experiment_name} seed {seed_path[-1]} all synthesizable molecules (N={len(all_rxn_steps)}) - # reaction steps: {round(np.mean(all_rxn_steps), 2)} ± {round(np.std(all_rxn_steps), 2)}")
        
        # Sort each reaction class by oracle calls
        for rxn_class_name in rxn_stats:
            rxn_stats[rxn_class_name] = sorted(rxn_stats[rxn_class_name])

        # Sort reaction classes by count in descending order
        sorted_stats = sorted(rxn_stats.items(), key=lambda x: len(x[1]), reverse=True)

        # Filter for reactions with count > 500 and take top 10
        sorted_stats = [(k, v) for k, v in sorted_stats if len(v) > 500][:10]

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
        
        plt.savefig(os.path.join(save_dir, f"{experiment_name}-seed{seed_path[-1]}-rxn-evolution.png"))
        plt.close()

def count_rxn_graph(top_graphs: Dict[str, Union[str, float]]) -> Union[Dict[str, int], List[int]]:
    """
    Count number of reaction classes and number of reaction steps for each top graph.
    """
    rxn_count = dict()
    rxn_steps = []

    for key, value in top_graphs.items():

        synthesis_graph = {k:v for k,v in top_graphs[key]["synthesis_data"].items() if "node" in k}
        steps = 0

        for node, attributes in synthesis_graph.items():
            if attributes["is_rxn"]:
                rxn_name = attributes["rxn_name"]
                if rxn_name not in rxn_count:
                    rxn_count[rxn_name] = 1
                else:
                    rxn_count[rxn_name] += 1
                    
                steps += 1

        rxn_steps.append(steps)
    
    return rxn_count, rxn_steps

def plot_top_graphs_rxn_classes(
    rxn_count: Dict[str, int],
    save_dir: str,
    experiment_name: str
) -> None:
    """Plot reaction class distribution and save barplot."""
    # Save the raw counts
    with open(os.path.join(save_dir, f"{experiment_name}-top-graphs-rxn-distribution.json"), "w") as f:
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

    plt.savefig(os.path.join(save_dir, f"{experiment_name}-top-graphs-rxn-distribution.png"))
    plt.close()

def annotate_rxn_conditions(
    top_graphs: Dict[str, Dict[str, Union[str, int]]],
    reacon_dir: str
) -> Dict[str, Dict[str, Union[str, int]]]:
    """
    Annotate the conditions for the top graphs using Reacon: https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc05946h#cit27.
    """
    # Loop through top_graphs and extract all reaction nodes
    reaction_smiles = []
    for smiles, data in top_graphs.items():
        synthesis_graph = {k:v for k,v in data["synthesis_data"].items() if "node" in k}
        for node, attributes in synthesis_graph.items():
            if attributes["is_rxn"]:
                reaction_smiles.append(attributes["rxn_smiles"])

    # Write a temporary DataFrame out following the required format
    df = pd.DataFrame({
        "_id": list(range(len(reaction_smiles))),  # Dummy attribute
        "reaction_smiles": reaction_smiles
    })
    df.to_csv("temp_reaction_smiles.csv", index=False)

    # Run Reacon
    subprocess.run([
        "bash", 
        os.path.join(reacon_dir, "map_and_pred_conditions.sh"),
        os.path.abspath(os.path.dirname(__file__)),
        reacon_dir
    ])

    # Extract the cluster predictions
    conditions = []
    reacon_output = json.load(open("temp_predictions/cluster_condition_prediction.json"))
    for rxn_id, preds in reacon_output.items():
        top_1_condition = preds[1][0]["best condition"]
        conditions.append(top_1_condition)

    # Add conditions to top_graphs
    condition_idx = 0
    for smiles, data in top_graphs.items():
        synthesis_graph = {k:v for k,v in data["synthesis_data"].items() if "node" in k}
        for node, attributes in synthesis_graph.items():
            if attributes["is_rxn"]:
                condition_dict = {
                    "catalyst": conditions[condition_idx][0],
                    "solvent_1": conditions[condition_idx][1], 
                    "solvent_2": conditions[condition_idx][2],
                    "reagent_1": conditions[condition_idx][3],
                    "reagent_2": conditions[condition_idx][4], 
                    "reagent_3": conditions[condition_idx][5]
                }
                attributes["conditions"] = condition_dict
                condition_idx += 1

    # Remove temporary files
    os.remove("temp_reaction_smiles.csv")
    os.remove("temp_reaction_templates.csv")
    shutil.rmtree("temp_predictions")
    os.remove("sample_preds.csv")
    
    return top_graphs
