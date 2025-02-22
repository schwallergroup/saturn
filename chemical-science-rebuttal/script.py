import os
import subprocess
import time

start = time.perf_counter()

BASE_PATH = "/home/jeff/saturn-dev/chemical-science-rebuttal"

for exp in ["saturn-all-mpo-no-aizynth", "saturn-just-docking"]:
    for seed in range(10):
        experiment_base_path = os.path.join(BASE_PATH, exp)
        full_config_path = os.path.join(experiment_base_path, f"config_{seed}.json")
        subprocess.run([
            "python", 
            "/home/jeff/saturn-dev/saturn.py", 
            full_config_path
        ])
        # Rename oracle_history.csv output
        subprocess.run([
            "mv",
            os.path.join(experiment_base_path, "oracle_history.csv"),
            os.path.join(experiment_base_path, f"oracle_history_{seed}.csv")
        ])

end = time.perf_counter()
print(f"Finished in {end-start:.2f} seconds")
