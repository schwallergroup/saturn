"""
Script to extract and plot metrics from TANGO-RXN runs (post-run).
"""
from typing import List, Tuple
import os
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
    count_rxn_graph,
    plot_rxn_classes
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
    if docking_oracle == "quickvina2":
        return QUICKVINA2_DOCKING_SCORE_ENUM
    elif docking_oracle == "gnina":
        return GNINA_DOCKING_SCORE_ENUM
    else:
        raise ValueError(f"Docking oracle {docking_oracle} not supported")

# Fixed utility script paths
SATURN_BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROUTE_EXTRACTION_SCRIPT = os.path.join(SATURN_BASE_PATH, "oracles/synthesizability/utils/extract_syntheseus_route_data.py")
RXN_INFO_EXTRACTION_SCRIPT = os.path.join(SATURN_BASE_PATH, "oracles/synthesizability/utils/extract_rxn_info.py")


def log_num_solved(seeds: List[str]) -> None:
    """Log number of solved molecules to logging file"""
    N = 0
    non_solved = []
    solved = []
    
    for seed in seeds:
        df = pd.read_csv(os.path.join(seed, "oracle_history.csv"))

        # Non-solved molecules
        df_non_solved = df.loc[df[SYNTHESEUS_REWARD_ENUM] == 0]
        try: 
            df_non_solved = df_remove_duplicate_smiles(df_non_solved)
        except Exception:
            logging.info("Error in *non-solved* de-duplication")
        non_solved.append(len(df_non_solved))

        # Solved molecules
        df_solved = df.loc[df[SYNTHESEUS_REWARD_ENUM] == 1]
        try:
            df_solved = df_remove_duplicate_smiles(df_solved)
        except Exception:
            logging.info("Error in *solved* de-duplication")
        solved.append(len(df_solved))

        # Number of successes
        if len(df_solved) > 0:
            N += 1

    if len(solved) > 0:
        logging.info(f"Successful Runs: {N}, Non-solved: {int(np.mean(non_solved))} ± {int(np.std(non_solved))}, Solved: {int(np.mean(solved))} ± {int(np.std(solved))}")
        logging.info(f"Raw Non-solved: {non_solved}, Raw Solved: {solved}")
    else:
        logging.info(f"No runs generated any solved molecules.")


def log_wall_time(seeds: List[str]) -> None:
    """Log run wall time"""
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


def log_pooled_molecules_stats(
    pooled_data: List[Tuple[str, float, float]],  # (SMILES, docking_score, qed)
    threshold_label: str
) -> None:
    """Log pooled molecules stats"""
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

        logging.info(f"{threshold_label}: {np.mean(docking_scores):.2f} ± {np.std(docking_scores):.2f} QED: {np.mean(qed_values):.2f} ± {np.std(qed_values):.2f} (N={len(pooled_data)})")
        logging.info(f"Ligand Efficiency: {np.mean(ligand_efficiency_values):.2f} ± {np.std(ligand_efficiency_values):.2f}, # Unique Bemis-Murcko Scaffolds: {scaffolds}")
        logging.info(f"IntDiv1: {intdiv1:.3f}, #Circles (T=0.75): {circles_high}, #Circles (T=0.50): {circles_low}\n")

    else:
        logging.info(f"No molecules generated for {threshold_label} docking score interval")


def log_molecule_and_rxn_stats(
    seeds: List[str],    
    experiment_name: str,
    save_top_graphs: bool = True,
    save_top_percentage_routes: float = 0.01,
    save_dir: str = "./top_graphs/"
) -> None:
    """Log molecule and reaction metrics stats"""
    # If top graphs should be saved 
    if save_top_graphs:
        top_oracle_histories = [] # Store DataFrames with top molecules

    N_rxn = 0
    num_enforced_rxn = []
    enforced_rxn_metrics = []  # List of (SMILES, docking_score, qed) tuples
    top_enforced_rxn_metrics = []

    for seed in seeds:
        enforce_reactions, enforce_building_blocks, enforced_building_blocks_file = get_run_data(seed)
        assert enforce_reactions, "Reaction presence was not enforced"

        # Skip if oracle_history.csv doesn't exist (failed run)
        if not os.path.exists(os.path.join(seed, "oracle_history.csv")):
            continue
            
        # Load and canonicalize SMILES from JSON files
        rxn_json = json.load(open(os.path.join( 
                                seed, 
                                "syntheseus_results", 
                                "matched_generated_smiles_with_rxn.json"), 
                                "r"
                            )
                        )
        
        enforced_rxn_smiles = set([canonicalize_smiles(s) for smiles_list in rxn_json.values() for s in smiles_list if s])

        if len(enforced_rxn_smiles) > 0:
            N_rxn += 1

        # Load and filter DataFrame
        df = df_remove_duplicate_smiles(pd.read_csv(os.path.join(seed, "oracle_history.csv")))
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)
        
        # Track metrics
        num_enforced_rxn.append(len(enforced_rxn_smiles))
        df_enforced_rxn = df[df["canonical_smiles"].isin(enforced_rxn_smiles)]  # This keeps only syntheseus_raw_values == 1
        df_enforced_rxn = df_enforced_rxn.sort_values(by=REWARD_ENUM, ascending=False)
        df_top_enforced_rxn = df_enforced_rxn.head(int(save_top_percentage_routes * len(df_enforced_rxn)))

        assert abs(len(df_enforced_rxn) - len(enforced_rxn_smiles)) <= 10, f"Large mismatch in number of SMILES matched in DataFrame: {len(df_enforced_rxn)} != {len(enforced_rxn_smiles)}"

        enforced_rxn_metrics.extend(zip(
            df_enforced_rxn["canonical_smiles"], 
            df_enforced_rxn[DOCKING_SCORE_ENUM], 
            df_enforced_rxn[QED_ENUM]
        ))

        top_enforced_rxn_metrics.extend(zip(  # Top molecules (by reward, given syntheseus_raw_values == 1)
            df_top_enforced_rxn["canonical_smiles"], 
            df_top_enforced_rxn[DOCKING_SCORE_ENUM], 
            df_top_enforced_rxn[QED_ENUM]
        ))

        if save_top_graphs:
            top_oracle_histories.append((df_top_enforced_rxn, seed))


    logging.info(f"# Enforced Blocks (if applicable) and Reaction (N={N_rxn}): {int(np.mean(num_enforced_rxn))} ± {int(np.std(num_enforced_rxn))}\n")

    # Log the mean and std of the docking scores across thresholds (-8 to -9, -9 to -10, < -10)
    logging.info(f"Enforced Reactions Docking Score Stats:\n")

    # -8 to -9
    metrics_8_9 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if -9 <= dock < -8]
    log_pooled_molecules_stats(metrics_8_9, "Docking Scores: -8 to -9")
    
    # -9 to -10
    metrics_9_10 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if -10 <= dock < -9]
    log_pooled_molecules_stats(metrics_9_10, "Docking Scores: -9 to -10")
    
    # < -10
    metrics_10 = [(smiles, dock, qed) for smiles, dock, qed in enforced_rxn_metrics if dock < -10]
    log_pooled_molecules_stats(metrics_10, "Docking Scores: < -10")

    # Highest Reward
    metrics_highest_reward = [(smiles, dock, qed) for smiles, dock, qed in top_enforced_rxn_metrics]
    log_pooled_molecules_stats(metrics_highest_reward, f"Top {save_top_percentage_routes * 100}% by Reward")
    
    # If save_top_graphs, iterate over graphs, save them and plot reaction distribution
    if save_top_graphs:
        logging.info(f"Saving top {save_top_percentage_routes * 100}% of routes for experiment: {experiment_name}")

        top_graphs = {}

        # For each DataFrame, slice the top % and save routes for visualization
        for df, seed in top_oracle_histories:

            syntheseus_path = os.path.join(seed, "syntheseus_results")
            # TODO: Check why the number of solved molecules does not match with the one if syntheseus_raw_values == 1
            extracted_graph = write_out_top_syntheseus_graphs(
                oracle_history=df,
                syntheseus_folder=syntheseus_path,
                enforce_building_blocks=enforce_building_blocks,
                enforced_building_blocks_file=enforced_building_blocks_file,
                syntheseus_path_script=ROUTE_EXTRACTION_SCRIPT,
                rxn_info_path_script=RXN_INFO_EXTRACTION_SCRIPT
            )
        
            top_graphs.update(extracted_graph)
        
        # Save top graphs
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, f"{experiment_name}-top-graphs.json"), "w") as f:
            json.dump(top_graphs, f, indent=4)

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

        plot_rxn_classes(
            rxn_count=rxn_count,
            save_dir=save_dir,
            experiment_name=experiment_name
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run analysis of TANGO-RXN runs and output a log file of metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Names of the experiment folders to analyze."
    )
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="Path to parent directory containing the experiment folders."
    )
    parser.add_argument(
        "--save_top_graphs",
        action="store_true",
        default=True,
        help="Extract and save synthesis graphs for top molecules."
    )
    parser.add_argument(
        "--save_top_percentage_routes",
        type=float,
        default=0.005,
        help="Percentage of top routes to save (e.g. 0.005 for top 0.5%%)."
    )
    parser.add_argument(
        "--docking_oracle",
        type=str,
        required=True,
        choices=["quickvina2", "gnina"],
        help="Docking oracle used for scoring"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("./metrics.log")
    logging.info(f"Saving top graphs: {args.save_top_graphs} - If True, top {args.save_top_percentage_routes * 100}% of routes will be saved.")

    start_time = time.perf_counter()

    for experiment_name in args.experiments:
        experiment_start_time = time.perf_counter()
        logging.info(f"Starting analysis for experiment: {experiment_name}\n")

        path = os.path.join(args.experiment_path, experiment_name)
        seeds = [os.path.join(path, f"seed{seed}") for seed in range(5) if os.path.exists(os.path.join(path, f"seed{seed}"))]
        assert len(seeds) == 5, f"Expected 5 seeds, found {len(seeds)} for experiment {experiment_name}."

        logging.info(f"----- Results for: {experiment_name} -----")

        # Log number of solved molecules
        log_num_solved(seeds)

        # Log wall time
        log_wall_time(seeds)

        DOCKING_SCORE_ENUM = get_docking_score_enum(args.docking_oracle)

        # Log enforced reaction and building blocks info
        log_molecule_and_rxn_stats(
            seeds=seeds,
            experiment_name=experiment_name,
            save_top_graphs=args.save_top_graphs,
            save_top_percentage_routes=args.save_top_percentage_routes
        )

        experiment_end_time = time.perf_counter()
        logging.info(f"Experiment: {experiment_name} analysis time: {experiment_end_time - experiment_start_time:.2f} seconds\n")

    end_time = time.perf_counter()

    logging.info(f"Total analysis time: {end_time - start_time:.2f} seconds for {len(args.experiments)} experiments")
