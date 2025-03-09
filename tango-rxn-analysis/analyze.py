"""
Script to extract and plot metrics from TANGO-RXN runs (post-run).
"""
from typing import List, Tuple
import os
import subprocess
import time
import logging
import argparse
import json
import pandas as pd
import numpy as np
from rdkit import Chem

from utils import (
    setup_logging,
    df_remove_duplicate_smiles, 
    canonicalize_smiles, 
    get_wall_time,
    get_morgan_fingerprints,
    get_run_data,
    ligand_efficiency,
    num_unique_murcko_scaffolds,
    internal_diversity,
    NCircles,
    write_out_top_syntheseus_graphs,
    plot_rxn_evolution,
    count_rxn_graph,
    plot_top_graphs_rxn_classes,
    annotate_rxn_conditions
)

# -----------------
# Global Variables
# -----------------
pd.options.mode.chained_assignment = None

# Enums
REWARD_ENUM = "reward"
SYNTHESEUS_REWARD_ENUM = "syntheseus_reward"
QED_ENUM = "qed_raw_values"
HBD_ENUM = "num_hbd_raw_values"
QUICKVINA2_DOCKING_SCORE_ENUM = "quickvina2_gpu_raw_values"
GNINA_DOCKING_SCORE_ENUM = "gnina_raw_values"

def get_docking_score_enum(docking_oracle: str) -> str:
    if docking_oracle in ["quickvina", "quickvina2"]:
        return QUICKVINA2_DOCKING_SCORE_ENUM
    elif docking_oracle == "gnina":
        return GNINA_DOCKING_SCORE_ENUM
    else:
        raise ValueError(f"Docking oracle {docking_oracle} not supported")

def log_synthesizable_metrics(
    seeds_paths: List[str],
    minimize_path_length: bool
) -> None:
    """Log number of synthesizable molecules."""
    N = 0
    non_synthesizable = []
    synthesizable = []
    synthesizable_with_constraints = []
    
    for seed_path in seeds_paths:
        # Skip if oracle_history.csv doesn't exist (failed run)
        if not os.path.exists(os.path.join(seed_path, "oracle_history.csv")):
            continue

        df = pd.read_csv(os.path.join(seed_path, "oracle_history.csv"))
        # Number of generated molecules
        num_generated_molecules = len(df)

        # Synthesizable molecules
        syntheseus_results_dir = os.path.join(seed_path, "syntheseus_results")
        num_pkl = int(subprocess.check_output(f"find {syntheseus_results_dir} -type f -name '*.pkl' | wc -l", shell=True))
        synthesizable.append(num_pkl)

        # Non-synthesizable molecules
        non_synthesizable.append(num_generated_molecules - num_pkl)

        # Synthesizable molecules with all constraints
        if not minimize_path_length:
            df_synthesizable_with_constraints = df.loc[df[SYNTHESEUS_REWARD_ENUM] == 1]
            try:
                df_synthesizable_with_constraints = df_remove_duplicate_smiles(df_synthesizable_with_constraints)
            except Exception:
                logging.info("Error in *synthesizable with constraints* de-duplication")
            synthesizable_with_constraints.append(len(df_synthesizable_with_constraints))
        else:
            enforce_reactions, enforced_reaction_classes, enforce_building_blocks, enforced_building_blocks_file = get_run_data(seed_path)
            assert enforce_reactions, "Reaction presence was not enforced"
            # Load and canonicalize SMILES from JSON files
            smiles_rxn_tracker = json.load(open(os.path.join(seed_path, "syntheseus_results", "smiles_rxn_tracker.json"), "r"))

            # If also enforcing building blocks
            if enforce_building_blocks:
                count_with_enforced_block = sum(1 for rxn_info in smiles_rxn_tracker.values() if rxn_info["enforced_block"] is not None)
                synthesizable_with_constraints.append(count_with_enforced_block)
            else:
                # In case of duplicates
                matched_smiles = set()
                matched_rxn_count = 0
                for smiles, rxn_info in smiles_rxn_tracker.items():
                    canonical_smiles = canonicalize_smiles(smiles)
                    if canonical_smiles in matched_smiles:
                        continue
                    for v in rxn_info.values():
                        if (isinstance(v, dict)) and (v.get("rxn_class") is not None):
                            if any(rxn in v["rxn_class"].lower() or rxn in v["rxn_name"].lower() for rxn in enforced_reaction_classes):
                                matched_rxn_count += 1
                                matched_smiles.add(canonical_smiles)
                                break
                synthesizable_with_constraints.append(matched_rxn_count)
            
        # Number of successes
        if synthesizable_with_constraints[-1] > 0:
            N += 1

    if len(synthesizable_with_constraints) > 0:
        logging.info(f"# Successful Runs: {N}")
        logging.info(f"Non-synthesizable: {int(np.mean(non_synthesizable))} ± {int(np.std(non_synthesizable))}, Raw values: {non_synthesizable}")
        logging.info(f"Synthesizable (not necessarily with all constraints): {int(np.mean(synthesizable))} ± {int(np.std(synthesizable))}, Raw values: {synthesizable}")
        logging.info(f"Synthesizable (with all constraints): {int(np.mean(synthesizable_with_constraints))} ± {int(np.std(synthesizable_with_constraints))}, Raw values: {synthesizable_with_constraints}")
    else:
        logging.info(f"No runs generated any synthesizable molecules (with all constraints).")

def log_wall_time(seeds: List[str]) -> None:
    """Log run wall time."""
    times = []

    for seed in seeds:
        with open(os.path.join(seed, "log.log"), "r") as f:
            lines = f.readlines()
            try:
                times.append(get_wall_time(lines))  # Returned time is in seconds
            except Exception:
                logging.info(f"Error in wall time extraction for {seed}")

    if len(times) > 0:
        # Convert times to hours and minutes
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_hours, mean_minutes = divmod(mean_time, 3600)
        mean_minutes = mean_minutes / 60
        std_hours, std_minutes = divmod(std_time, 3600)
        std_minutes = std_minutes / 60
        logging.info(f"Wall Time (N={len(times)}): {int(mean_hours)}h {int(mean_minutes)}m ± {int(std_hours)}h {int(std_minutes)}m")

def log_pooled_molecules_metrics(
    pooled_data: List[Tuple[str, float, float]],  # (SMILES, docking_score, qed)
    threshold_label: str
) -> None:
    """Log pooled molecules metrics."""
    if pooled_data:
        docking_scores = [dock for _, dock, _ in pooled_data]
        qed_values = [qed for _, _, qed in pooled_data]
        # Isolate SMILES
        smiles = [smiles for smiles, _, _ in pooled_data]
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles]
        # Filter None
        mols = [mol for mol in mols if mol is not None]
        assert len(mols) == len(smiles), "Mismatch in number of molecules and SMILES strings"

        ligand_efficiency_values = ligand_efficiency(mols, docking_scores)
        # Diversity metrics
        # Scaffolds
        scaffolds = num_unique_murcko_scaffolds(mols)
        intdiv1 = internal_diversity(mols)
        # Get fingerprints
        fps = get_morgan_fingerprints(mols, as_list=True)
        circles_high = NCircles(threshold=0.75).measure(fps)
        circles_low = NCircles(threshold=0.50).measure(fps)

        logging.info(f"{threshold_label}: {np.mean(docking_scores):.2f} ± {np.std(docking_scores):.2f} QED: {np.mean(qed_values):.2f} ± {np.std(qed_values):.2f}")
        logging.info(f"Ligand Efficiency: {np.mean(ligand_efficiency_values):.2f} ± {np.std(ligand_efficiency_values):.2f}, # Unique Bemis-Murcko Scaffolds: {scaffolds}")
        logging.info(f"IntDiv1: {intdiv1:.3f}, #Circles (T=0.75): {circles_high}, #Circles (T=0.50): {circles_low}\n")

    else:
        logging.info(f"No molecules generated for {threshold_label} docking score interval")

def log_molecule_and_rxn_metrics(
    seeds_paths: List[str],    
    minimize_path_length: bool,
    experiment_path: str,
    experiment_name: str,
    save_top_percentage_routes: float,
    save_dir: str = "./top_graphs/"
) -> None:
    """Log molecule and reaction metrics."""

    # Make save directory
    os.makedirs(save_dir, exist_ok=True)

    # Store DataFrames with top molecules
    top_oracle_histories = []

    N_rxn = 0
    num_enforced_rxn = []
    enforced_rxn_metrics = []  # List of (SMILES, docking_score, QED) tuples
    top_enforced_rxn_metrics = []

    # Store per-seed results
    seed_results = []

    for seed_path in seeds_paths:
        enforce_reactions, enforced_reaction_classes, enforce_building_blocks, enforced_building_blocks_file = get_run_data(seed_path)
        assert enforce_reactions, "Reaction presence was not enforced"

        # Skip if oracle_history.csv doesn't exist (failed run)
        if not os.path.exists(os.path.join(seed_path, "oracle_history.csv")):
            continue
            
        # Load and canonicalize SMILES from JSON files
        rxn_json = json.load(open(os.path.join(seed_path, "syntheseus_results", "matched_generated_smiles_with_rxn.json"), "r"))
        enforced_rxn_smiles = set([canonicalize_smiles(s) for smiles_list in rxn_json.values() for s in smiles_list if s])

        if len(enforced_rxn_smiles) > 0:
            N_rxn += 1

        # Load and filter DataFrame
        df = df_remove_duplicate_smiles(pd.read_csv(os.path.join(seed_path, "oracle_history.csv")))
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        
        # Track metrics
        num_enforced_rxn.append(len(enforced_rxn_smiles))

        df_enforced_rxn = df[df["canonical_smiles"].isin(enforced_rxn_smiles)]  
        if not minimize_path_length:
            # Double check that all matched molecules have syntheseus_raw_values == 1
            assert sum(df_enforced_rxn[SYNTHESEUS_REWARD_ENUM]) == len(df_enforced_rxn), "Error: Not all matched molecules from the JSON have syntheseus_raw_values == 1 in the oracle history."

        df_enforced_rxn = df_enforced_rxn.sort_values(by=REWARD_ENUM, ascending=False)
        # Get the top molecules
        df_top_enforced_rxn = df_enforced_rxn.head(int(save_top_percentage_routes * len(df_enforced_rxn)))

        assert abs(len(df_enforced_rxn) - len(enforced_rxn_smiles)) <= 10, f"Large mismatch in number of SMILES matched in DataFrame: {len(df_enforced_rxn)} != {len(enforced_rxn_smiles)}"

        # Store this seed's results
        seed_metrics = list(zip(
            df_enforced_rxn["canonical_smiles"],
            df_enforced_rxn[DOCKING_SCORE_ENUM],
            df_enforced_rxn[QED_ENUM]
        ))
        seed_results.append(seed_metrics)

        # Add to pooled results
        enforced_rxn_metrics.extend(seed_metrics)

        top_enforced_rxn_metrics.extend(zip(  # Top molecules (by reward, given syntheseus_raw_values == 1)
            df_top_enforced_rxn["canonical_smiles"], 
            df_top_enforced_rxn[DOCKING_SCORE_ENUM], 
            df_top_enforced_rxn[QED_ENUM]
        ))

        # Store top molecules
        top_oracle_histories.append((df_top_enforced_rxn, seed_path))

    logging.info(f"# Enforced Blocks (if applicable) and Reaction (N={N_rxn}): {int(np.mean(num_enforced_rxn))} ± {int(np.std(num_enforced_rxn))}\n")

    # Log pooled metrics with per-seed molecule counts in parentheses
    # -8 to -9
    metrics_8_9 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if -9 <= dock < -8]
    seed_counts_8_9 = [len([(smiles, dock, qed) for smiles, dock, qed in seed_metrics if -9 <= dock < -8]) for seed_metrics in seed_results]
    seed_counts_str_8_9 = [str(count) for count in seed_counts_8_9]
    log_pooled_molecules_metrics(metrics_8_9, f"Docking Scores: -8 to -9 (N={len(metrics_8_9)}, per seed: {', '.join(seed_counts_str_8_9)}, Mean ± Std: {int(np.mean(seed_counts_8_9))} ± {int(np.std(seed_counts_8_9))})")
    
    # -9 to -10
    metrics_9_10 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if -10 <= dock < -9]
    seed_counts_9_10 = [len([(smiles, dock, qed) for smiles, dock, qed in seed_metrics if -10 <= dock < -9]) for seed_metrics in seed_results]
    seed_counts_str_9_10 = [str(count) for count in seed_counts_9_10]
    log_pooled_molecules_metrics(metrics_9_10, f"Docking Scores: -9 to -10 (N={len(metrics_9_10)}, per seed: {', '.join(seed_counts_str_9_10)}, Mean ± Std: {int(np.mean(seed_counts_9_10))} ± {int(np.std(seed_counts_9_10))})")
    
    # < -10
    metrics_10 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if dock < -10]
    seed_counts_10 = [len([(smiles, dock, qed) for smiles, dock, qed in seed_metrics if dock < -10]) for seed_metrics in seed_results]
    seed_counts_str_10 = [str(count) for count in seed_counts_10]
    log_pooled_molecules_metrics(metrics_10, f"Docking Scores: < -10 (N={len(metrics_10)}, per seed: {', '.join(seed_counts_str_10)}, Mean ± Std: {int(np.mean(seed_counts_10))} ± {int(np.std(seed_counts_10))})")

    # Highest Reward
    metrics_highest_reward = [(smiles, dock, qed) for smiles, dock, qed in top_enforced_rxn_metrics]
    log_pooled_molecules_metrics(metrics_highest_reward, f"Top {save_top_percentage_routes * 100}% by Reward (N={len(metrics_highest_reward)}), Docking Scores:")

    # Plot evolution of reaction classes
    plot_rxn_evolution(
        seeds_paths=seeds_paths,
        enforced_rxn=experiment_name,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Iterate over top molecules and save routes for visualization
    logging.info(f"Saving top {save_top_percentage_routes * 100}% of routes for experiment: {experiment_name}")

    top_graphs = {}

    # For each DataFrame, slice the top % and save routes for visualization
    for df, seed_path in top_oracle_histories:

        smiles_rxn_tracker = json.load(open(os.path.join(seed_path, "syntheseus_results", "smiles_rxn_tracker.json"), "r"))

        extracted_graph = write_out_top_syntheseus_graphs(
            top_oracle_history=df,
            smiles_rxn_tracker=smiles_rxn_tracker
        )
    
        top_graphs.update(extracted_graph)

    # Count number of unique enforced blocks amongst top graphs
    if enforce_building_blocks:
        unique_enforced_blocks = set()
        for graph in top_graphs.values():
            unique_enforced_blocks.add(graph["enforced_block"])

        total_num_enforced_blocks = len(set([canonicalize_smiles(s) for s in open(enforced_building_blocks_file).readlines()]))
        logging.info(f"Unique Enforced Blocks: {len(unique_enforced_blocks)}/{total_num_enforced_blocks}")

    # Plot reactions
    rxn_count, rxn_steps = count_rxn_graph(top_graphs)
    logging.info(f"Top Graphs Reaction Steps: {np.mean(rxn_steps):.2f} ± {np.std(rxn_steps):.2f}")

    plot_top_graphs_rxn_classes(
        rxn_count=rxn_count,
        save_dir=save_dir,
        experiment_name=experiment_name
    )

    # Annotate reaction conditions
    logging.info(f"Annotating reaction conditions for the top graphs for experiment: {experiment_name}")
    try:
        top_graphs = annotate_rxn_conditions(
            top_graphs=top_graphs,
            reacon_dir="/home/jeff/saturn-dev/test/testing-reacon/reacon"
        )
    except Exception:
        logging.info(f"Error in reaction condition annotation for experiment: {experiment_name}. Do not expect any reaction conditions to be annotated for the top graphs.")

    with open(os.path.join(save_dir, f"{experiment_name}-top-graphs.json"), "w") as f:
        json.dump(top_graphs, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis of TANGO-RXN runs and output a log file of metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to parent directory containing the experiment folders."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Names of the experiment folders to analyze."
    )
    parser.add_argument(
        "--docking_oracle",
        type=str,
        required=True,
        choices=["quickvina", "quickvina2", "gnina"],
        help="Docking oracle used in the reward function."
    )
    parser.add_argument(
        "--save_top_percentage_routes",
        type=float,
        default=0.005,
        help="Percentage of top routes to save (e.g. 0.005 for top 0.5%%)."
    )
    parser.add_argument(
        "--minimize_path_length",
        action="store_true",
        help="Whether the experiment minimizes the path length of the synthetic routes."
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging("./metrics.log")
    logging.info(f"The script will save the top {args.save_top_percentage_routes * 100}% of routes for each experiment.")
    
    start_time = time.perf_counter()

    for experiment_name in args.experiments:
        experiment_start_time = time.perf_counter()
        logging.info(f"Starting analysis for experiment: {experiment_name}\n")

        path = os.path.join(args.experiment_path, experiment_name)
        seeds_paths = [os.path.join(path, folder) for folder in os.listdir(path) if folder.startswith("seed") and os.path.isdir(os.path.join(path, folder))]
        assert len(seeds_paths) > 0, f"No seeds found for experiment {experiment_name}."
        # FIXME: Some parts of the code assume that the format of seed_paths ends with */seed{idx}
        seeds_paths.sort()

        logging.info(f"----- Results for: {experiment_name} -----")

        # Log number of solved molecules
        log_synthesizable_metrics(
            seeds_paths=seeds_paths,
            minimize_path_length=args.minimize_path_length
        )

        # Log wall time
        log_wall_time(seeds_paths)

        DOCKING_SCORE_ENUM = get_docking_score_enum(args.docking_oracle)

        # Log enforced reaction and building blocks info
        log_molecule_and_rxn_metrics(
            seeds_paths=seeds_paths,
            minimize_path_length=args.minimize_path_length,
            experiment_path=args.experiment_path,
            experiment_name=experiment_name,
            save_top_percentage_routes=args.save_top_percentage_routes
        )

        experiment_end_time = time.perf_counter()
        logging.info(f"Experiment: {experiment_name} analysis time: {experiment_end_time - experiment_start_time:.2f} seconds\n")

    end_time = time.perf_counter()

    logging.info(f"Total analysis time: {end_time - start_time:.2f} seconds for {len(args.experiments)} experiments")
