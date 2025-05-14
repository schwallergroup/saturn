# Instructions for Reproducing the Synthesizability Control Experiments 

[Pre-print](https://arxiv.org/abs/2505.08774)

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


Rxn-INSIGHT (Open-source Reaction Labeling)
------------------------------------------
1. Clone the [repository](https://github.com/schwallergroup/Rxn-INSIGHT) - we used a forked version of the original repo 
2. Create the conda environment: `conda env create -f environment.yml`
3. Activate the environment: `conda activate rxn-insight`
4. Install the package: `pip install -e .` 


NameRxn (Proprietary Reaction Labeling Tool from NextMove Software)
-------------------------------------------------------------------
**NOTE**: This installation is optional - `NameRxn` is a proprietary tool requiring a license. Both `HazELNut` and `NameRxn` are required and can be downloaded from this link: https://www.nextmovesoftware.com/downloads/hazelnut/releases/. The exact version can be chosen (or download the latest). After downloading, extract and verify the tool works by running the following command (in the directory the tools were downloaded):

`./HazELNut/namerxn`

This should return the program interface.


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


Gnina (Docking used in Waste Valorization Experiments)
------------------------------------------------------
The developers of `Gnina` offer pre-built binaries. Download a compatible one from: https://github.com/gnina/gnina/releases. Alternatively, instructions are provided in the codebase to build the binary, if desired.


Unit Tests
------------------------------------------------------
Check that the reaction constraints are properly working with the units tests located in `<path to saturn>/tests/oracles/syntheseus/test_syntheseus.py`.

**NOTE**: The `NameRxn` binary executable path needs to be specified in the unit test file. If not using `NameRxn`, note that currently some tests will fail. In the future, `Rxn-INSIGHT` and `NameRxn` unit tests will be disentangled.

With the `syntheseus-full` environment activated:

`conda activate syntheseus-full`

Change directory to `<path to saturn>/tests/oracles/syntheseus` and run:

`pytest test_syntheseus.py`


Synthesizability Control Experiments
---------------------------------------------------------------------------

The raw data and building blocks stocks used in the pre-print are available [here](https://doi.org/10.6084/m9.figshare.29040977.v1). In the pre-print, many reaction constraints were imposed. Instead of providing individual run configurations (as `JSONs`), we thought it would be most practical to provide a minimal Jupyter notebook demonstrating how to impose arbitrary constraints. This way, if interested, the framework can be adapted to user-specific case studies. 


`./tutorial.ipynb` is a minimal tutorial to generate configuration `JSONs` with arbitrary reaction constraints. Running the notebook requires `jupyter notebook` installed in the environment. If the above installation instructions were followed, the `rxn-insight` environment can be used to run the notebook. Otherwise, one can use any other environments as no specific packages are required. We then re-iterate that all experiments can be run as follows:

With the `saturn` environment activated:

`conda activate saturn`

all experiments can be run by passing the `JSON` configuration: 

`python saturn.py` <`JSON path`>
