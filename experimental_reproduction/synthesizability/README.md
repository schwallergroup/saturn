# Instructions for Reproducing the Synthesizability Experiments 

The **Saturn** environment can be installed via:

`source setup.sh`

**NOTE**: For all provided files, the paths to the `Saturn` repository and certain arguments needs to be changed in the provided configuration `JSONs`. This is indicated by `"< >"`.

The prepared files run the `Mamba` backbone trained on `ChEMBL 33`. Other model checkpoints are located in `checkpoint_models`. To run a different model, change the `prior` and `agent` paths in the `JSONs`.

With the `saturn` environment activated:

`conda activate saturn`

all experiments can be run by passing the `JSON` configuration: 

`python saturn.py` <`JSON path`>

Preliminary Installations
-------------------------

Experiment 1 does not require AiZynthFinder but Experiments 2 and 3 do. All cloning operations assume the current working directory is `./synthesizability`. This does not need to be the case, but some paths in the provided configuration `JSONs` assume this. To install at your preferred location, change the paths in the `JSONs`.

AiZynthFinder
-------------
1. Clone the [repository](https://github.com/MolecularAI/aizynthfinder)
2. Follow the installation instructions in the repository's README.md


Syntheseus
-------------
1. Clone the [repository](https://github.com/microsoft/syntheseus)
2. Created the conda full environment: `conda env create -f environment_full.yml`
3. Activate the environment: `conda activate syntheseus-full`
4. Install the package: `pip install -e ".[all]"`

**Potential Issues:**
1. **graphviz.backend.execute.ExecutableNotFound**:`sudo apt-get install graphviz` (***requires sudo permissions***)
2. **RetroKNN incompatible torch_scatter version**: `conda install pytorch-scatter -c pyg -c conda-forge` (***check cuda version***)

QuickVina2-GPU-2.1
------------------
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


Experiment 1: Optimizing only docking score leads to unreasonable molecules
---------------------------------------------------------------------------

This experiment illustrates how the QuickVina2-GPU-2.1 oracle can be exploited.


Experiment 2: Directly optimizing AiZynthFinder
-----------------------------------------------

Configuration files are provided for both `All MPO` and `Double MPO` objective functions.


Experiment 3: Directly optimizing AiZynthFinder starting from an unsuitable training distribution
--------------------------------------------------------------------------------------------------

This experiment uses essentially the same configuration files as Experiment 2. The only paths that need to be changed are the `prior` and `agent` to the `"purged" models`.


Experiment 4: Directly optimizing *any* Retrosynthesis Model
--------------------------------------------------------------------------------------------------

This experiment uses a similar configuration file to Experiment 2. The only difference is that the `AiZynthFinder` oracle is replaced with a `Syntheseus` oracle. The `building_blocks_file` and other relevant arguments need to be specified.

**Note**: To run different retrosynthesis models, change the `reaction_model` argument in the `Syntheseus` oracle. The current supported models are `RetroKNN`, `Graph2Edits`, and `RootAligned`.
