# Instructions for Reproducing the Synthesizability with Reaction Condition Constraints Experiments 

The **Saturn** environment can be installed via:

`source setup.sh`

**NOTE**: For all provided files, the paths to the `Saturn` repository and certain arguments needs to be changed in the provided configuration `JSONs`. This is indicated by `"< >"`.

The prepared file runs the `Mamba` backbone trained on `PubChem`. Other model checkpoints are located in `checkpoint_models`. To run a different model, change the `prior` and `agent` paths in the `JSONs`.

With the `saturn` environment activated:

`conda activate saturn`

all experiments can be run by passing the `JSON` configuration: 

`python saturn.py` <`JSON path`>

Preliminary Installations
-------------------------

Syntheseus (Retrosynthesis Framework)
-------------------------------------
1. Clone the [repository](https://github.com/microsoft/syntheseus)
2. Create the conda full environment: `conda env create -f environment_full.yml`
3. Activate the environment: `conda activate syntheseus-full`
4. Install the package: `pip install -e ".[all]"` 

**NOTE**: The last command installs all retrosynthesis models supported in `Syntheseus`. If only specific models are to be used, there is the option to install only those dependencies. See https://microsoft.github.io/syntheseus/stable/installation/.


NameRxn (Proprietary Reaction Labeling Tool from NextMove Software)
-------------------------------------------------------------------
**NOTE**: This installation is mandatory for reproducing the original experiments (QUARC was built using `NameRxn`) - `NameRxn` is a proprietary tool requiring a license. Both `HazELNut` and `NameRxn` are required and can be downloaded from this link: https://www.nextmovesoftware.com/downloads/hazelnut/releases/. The exact version can be chosen (or download the latest). After downloading, extract and verify the tool works by running the following command (in the directory the tools were downloaded):

`./HazELNut/namerxn`

This should return the program interface.

QUARC (Reaction condition annotation tool)
-------------------------------------------------------------------
Go to the [QUARC](https://github.com/coleygroup/quarc) repo and follow the Quick Start installation instructions. We used the version from commit `88cce64`. You would additionally need the `Pistachio Reaction Types.csv` file (available upon request from the authors of `QUARC`). 

**NOTE**: We use the option with `NameRxn` integration, although a full open-source version is also available.


QuickVina2-GPU-2.1 (GPU-accelerated Docking)
--------------------------------------------
**Install `Boost`**
1. `sudo apt-get install libboost-all-dev` (***requires sudo permissions***)
2. Download [Boost](https://www.boost.org/users/history/version_1_85_0.html): `wget https://archives.boost.io/release/1.85.0/source/boost_1_85_0.tar.gz`
3. `tar -xzf boost_1_85_0.tar.gz`
4. `cd boost_1_85_0`
5. `./bootstrap.sh --prefix=/usr/local`
6. `./b2 install --prefix=/usr/local --with=all -j$(nproc)`

**Install `QuickVina2-GPU-2.1`**
1. Clone the [repository](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1)
2. `cd QuickVina2-GPU-2.1`
3. `vim Makefile` and change paths
4. `make clean`
5. `make source`
6. Test the installation was successful by running `./QuickVina2-GPU-2-1 --config input_file_example/2bm2_config.txt` (change the `receptor` and `opencl_binary_path` paths in `input_file_example/2bm2_config.txt` first)
7. Re-install for faster implementation: `make clean` followed by `make`

**Install `clinfo`**

Certain arguments in `Makefile` requires information about the GPU. If the default value provided by the authors do not work, then `clinfo` can provide the relevant information. Install via: 

`sudo apt install clinfo`

Afterwards, running:

`clinfo`

will display NVIDIA information.


Reaction Condition Synthesizability Control Experiments
---------------------------------------------------------------------------

The raw data and building blocks stocks used in the pre-print are available [here](https://doi.org/10.6084/m9.figshare.29040977.v1). In the pre-print, many reaction constraints were imposed. Instead of providing individual run configurations (as `JSONs`), we thought it would be most practical to provide a minimal Jupyter notebook demonstrating how to impose arbitrary constraints. This way, if interested, the framework can be adapted to user-specific case studies. 


`./tutorial.ipynb` is a minimal tutorial to generate configuration `JSONs` with arbitrary reaction condition constraints. Running the notebook requires `jupyter notebook` installed in the environment. If the above installation instructions were followed, the `quarc` environment can be used to run the notebook. Otherwise, one can use any other environments as no specific packages are required. We then re-iterate that all experiments can be run as follows:

With the `saturn` environment activated:

`conda activate saturn`

all experiments can be run by passing the `JSON` configuration: 

`python saturn.py` <`JSON path`>

The protein crystal structure prepared for docking used in the development experiments (PDB: 7UVU) is available at `experimental_reproduction/synthesizability/7uvu-2-monomers-pdbfixer.pdbqt`, as well as the original ligand `experimental_reproduction/synthesizability/7uvu-reference.pdb`

The protein crystal structure prepared for docking used in the fungicide experiments (PDB: 9KQ3) is available at `experimental_reproduction/reaction_conditions_synthesizability/9KQ3.pdbqt`, as well as the original ligand `experimental_reproduction/reaction_conditions_synthesizability/9KQ3_ligand.pdb`