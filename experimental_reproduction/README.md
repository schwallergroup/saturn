# Instructions for Reproducing the Experiments

The **Saturn** environment can be installed via:
`source setup.sh`

Next, each sub-folder here contains prepared files to reproduce the experiments. With the `saturn` environment activated, all experiments can be run via `python saturn.py <config JSON>`.

**NOTE**: The path to the `Saturn` repository needs to be changed in the provided configuration `JSONs`. This is indicated by `"< >"`.

**NOTE**: The prepared files run the `Mamba` backbone. It *can* be run on CPU only but the wall time increases significantly. Therefore, we recommend using a GPU always, except for the Part 1 test experiment which will still run fast. To run on CPU, change `device` from **"cuda"** to **"cpu"** in the config `JSON`.

Part 1: Elucidating the Optimization Dynamics of Saturn
-------------------------------------------------------

This is the test experiment and the prepared configuration `JSON` can be directly run. Example output files are in `part-1/example-output-files`. `oracle_history.csv` contains all the generated molecules with their oracle component scores.


Part 2: Transferability of Sample Efficiency to Physics-based Oracles
---------------------------------------------------------------------

**NOTE**: Reproducing this experiment requires `DockStream` to perform `AutoDock Vina` docking. It can be found here: https://github.com/MolecularAI/DockStream. Clone the repository and install the conda environment using the `environment.yml` AutoDock Vina can be downloaded here: https://vina.scripps.edu/downloads/. The experiments were run on a Linux machine so the autodock_vina_1_1_2_linux_x86.tgz file was downloaded.

Docking was performed against 3 targets: `6cm4`, `3kc3`, `1eve`. The docking grid files and configuration `JSONs` (the actual Saturn `JSON` and the docking `JSON`) are provided in each corresponding sub-folder. The prepared files make the assumption that docking will be parallelized on 16 CPU cores. Since the batch size is 16, going beyond 16 cores will yield benefit and 16 cores *may* also not be optimal due to the overhead of parallelizing. To parallelize over fewer CPU cores, change the `number_cores` parameter in the docking `JSON`.


Part 3: Benchmarking Saturn
---------------------------

**NOTE**: This is the only experiment where the oracle budget is 3,000 instead of 1,000. This is for comparison to previous works, so we follow exactly the experimental protocol of GEAM: https://openreview.net/forum?id=sLGliHckR8.

**NOTE**: The following step may be required to enable QuickVina 2 execution. On terminal, change directory to `saturn/oracles/docking/docking_grids` and execute `chmod u+x qvina02`. This experiment also uses the ZINC 250k pre-trained prior (following the protocol from GEAM) instead of ChEMBL 33 for Part 1 and Part 2 experiments. 

Before the provided file can be run, some hard-coded paths need to be changed. These are located in `oracles/docking/geam_oracle.py`. The `self.vina_program` and `self.receptor_file` paths need to be changed to match your system.

Docking was performed against 5 targets: `parp1`, `fa7`, `5ht1b`, `braf`, `jak2`. The provided configuration `JSON` is for the `parp1` target. To change the target, change the `target` parameter. An error will be thrown if a target outside of these 5 are selected as we copy the code for the oracle directly from GEAM's code-base which only supports these targets. We copy to ensure exact comparison.
