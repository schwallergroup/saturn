import os
import json

BASE_PATH = "/home/jeff/saturn-dev/chemical-science-rebuttal"

for exp in ["saturn-all-mpo-no-aizynth", "saturn-just-docking"]:

    output_dir = os.path.join(BASE_PATH, exp)
    os.makedirs(output_dir, exist_ok=True)

    for seed in range(10):

        oracle_components = [
            {
                "name": "quickvina2_gpu",
                "weight": 1,
                "preliminary_check": False,
                "specific_parameters": {
                "binary": "/home/jeff/saturn-dev/experimental_reproduction/synthesizability/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1",
                "force_field": "uff", 
                "receptor": "/home/jeff/saturn-dev/experimental_reproduction/synthesizability/7uvu-2-monomers-pdbfixer.pdbqt",
                "reference_ligand": "/home/jeff/saturn-dev/experimental_reproduction/synthesizability/7uvu-reference.pdb",
                "thread": 8000,
                "results_dir": os.path.join(output_dir, f"docking_results_{seed}")
            },
            "reward_shaping_function_parameters": {
                "transformation_function": "reverse_sigmoid",
                "parameters": {
                    "low": -16,
                    "high": 0,
                    "k": 0.15
                }
            }
        }]

        if "all-mpo" in exp:
            oracle_components.append({
                "name": "qed",
                "weight": 1,
                "preliminary_check": False,
                "specific_parameters": {},
                "reward_shaping_function_parameters": {
                    "transformation_function": "no_transformation"
                }
            })
            oracle_components.append({
                "name": "sa_score",
                "weight": 1,
                "preliminary_check": False,
                "specific_parameters": {},
                "reward_shaping_function_parameters": {
                "transformation_function": "reverse_sigmoid",
                "parameters": {
                    "low": -1,
                    "high": 8,
                    "k": 0.25
                    }
                }
            })
    
        config = {
            "logging": {
                "logging_frequency": 1000000,
                "logging_path": os.path.join(output_dir, f"log_{seed}.log"),
                "model_checkpoints_dir": output_dir
            },
            "oracle": {
                "budget": 1000,
                "allow_oracle_repeats": False,
                "aggregator": "product",
                "components": oracle_components      
            },
            "goal_directed_generation": {
                "reinforcement_learning": {
                    "prior": "/home/jeff/saturn-dev/experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior",
                    "agent": "/home/jeff/saturn-dev/experimental_reproduction/checkpoint_models/zinc-250k-mamba-epoch-50.prior",
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "sigma": 128.0,
                    "augmented_memory": True,
                    "augmentation_rounds": 10,
                    "selective_memory_purge": True
                },
                "experience_replay": {
                    "memory_size": 100,
                    "sample_size": 10,
                    "smiles": []
                },
                "diversity_filter": {
                    "name": "IdenticalMurckoScaffold",
                    "bucket_size": 10
                },
                "hallucinated_memory": {
                    "execute_hallucinated_memory": False,
                    "hallucination_method": "ga",
                    "num_hallucinations": 100,
                    "num_selected": 5,
                    "selection_criterion": "random"
                },
                "beam_enumeration": {
                    "execute_beam_enumeration": False,
                    "beam_k": 2,
                    "beam_steps": 18,
                    "substructure_type": "structure",
                    "structure_min_size": 15,
                    "pool_size": 4,
                    "pool_saving_frequency": 1000,
                    "patience": 5,
                    "token_sampling_method": "topk",
                    "filter_patience_limit": 100000
                }
            },
            "distribution_learning": {
                "parameters": {
                    "agent": "<unused>",
                    "training_steps": 20,
                    "batch_size": 512,
                    "learning_rate": 0.0001,
                    "training_dataset_path": "cleaned-chembl-33.smi",
                    "train_with_randomization": True,
                    "transfer_learning": False
                }
            },
            "running_mode": "goal_directed_generation",
            "model_architecture": "mamba",
            "device": "cuda",
            "seed": seed
        }

        with open(os.path.join(output_dir, f"config_{seed}.json"), "w") as f:
            json.dump(config, f, indent=4)
