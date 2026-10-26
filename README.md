# Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation

<img src="saturn.jpeg" alt="Saturn Logo" width="300"/>

`Saturn` is a language model based molecular generative design framework that is focused on **sample-efficient *de novo* small molecule design**. 

In the **experimental_reproduction** sub-folder, prepared files and checkpoint models are provided to reproduce the experiments. 
There is also a `Jupyter` notebook to construct your own configuration files to run `Saturn`.

🚀 Installation
-------------

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Clone this Git repository
3. Open terminal and install the `saturn` environment. There are two options:

   **Option A — step-by-step install.** Run the installation script, which
   installs the same packages one by one (useful if you want to adapt
   individual versions; used historically in the development of the framework).:

        $ source setup.sh

   **Option B — pre-configured environment.** A ready-made,
   fully pinned export of the exact environment used for the paper is provided
   in [`environment.yml`](environment.yml). Recreate it in one command:

        $ conda env create -f environment.yml
        $ conda activate saturn

Potential Installation Issues
-----------------------------
* `GLIBCXX_3.4.29` version not found - thank you to [@PatWalters](https://github.com/PatWalters) for flagging this and solving via:

        $ conda uninstall openbabel 
        $ conda install gcc_linux-64
        $ conda install gxx_linux-64
        $ conda install -c conda-forge openbabel

* `ImportError: GLIBCXX_3.4.29 not found` - running the below commands may solve the issue:

        $ sudo add-apt-repository ppa:ubuntu-toolchain-r/test
        $ sudo apt update
        $ sudo apt install gcc-11 g++-11

* `causal-conv1d` and `mamba-ssm` installation error - see [Issue 1](https://github.com/schwallergroup/saturn/issues/1) - thank you to [@surendraphd](https://github.com/surendraphd) for sharing their solution.

⚙️ System Requirements
-------------------

* **Python** 3.10
* **GPU / CUDA:** a CUDA-enabled GPU with a driver compatible with **CUDA 11.3**. CPU-only execution also
  works but run times are much longer.
* **Operating system:** Linux. Tested on
  **Ubuntu 20.04 LTS**, **Ubuntu 24.04 LTS**, and **Red Hat Enterprise Linux 9.4**.
  Very recent distributions (e.g. Ubuntu 26.04) may conflict with the pinned dependencies.

<details>
<summary><b>ℹ️ A note on the pinned dependencies.</b></summary>
Saturn is the shared framework
underpinning a series of related works (see the References section). To keep results consistent and reproducible across all of these works, the framework deliberately pins a single, tested dependency stack (PyTorch, CUDA, Mamba, etc.) rather than tracking the latest releases. The exact versions are captured in [`environment.yml`](environment.yml); using the pre-configured environment above is the most reliable way to reproduce the reported results.
</details>

📋 Workflow
-------------------

Everything runs through a single entry point — `python saturn.py <config.json>` — where the `running_mode` field selects the phase. The high-level order of steps is: **1.** install (`setup.sh`) → **2.** get a prior (bundled checkpoint *or* `distribution_learning`) → **3.** build a config (`config_generator.ipynb`) → **4.** `goal_directed_generation` (sample → score → filter → memory manipulation → update, looped until the oracle budget runs out) → **5.** analyse the outputs.

Click below for the full flowchart mapping each step to the folder where its code lives, and the standalone navigation guide in [`WORKFLOW.md`](WORKFLOW.md).

<details>
<summary><b>📊 Click to expand the full workflow flowchart</b></summary>

```text
  STEP 1  ·  SETUP      Install the `saturn` conda environment
                        (+ external tools only if the oracle needs them:
                         DockStream / QuickVina2-GPU for docking;
                         Syntheseus / QUARC / NameRxn for retrosynthesis)
                        -> setup.sh , README.md, specific reproducibility README.md files
                                   |
                                   v
  STEP 2  ·  MODEL      Get a pre-trained prior (pick one):
                        A) reuse a bundled checkpoint
                           -> experimental_reproduction/checkpoint_models/ , models/
                        B) train your own  (running_mode: "distribution_learning")
                           -> distribution_learning/
                                   |
                                   v
  STEP 3  ·  CONFIG     Build the configuration JSON (objective, oracle
                        components, budget, memory settings)
                        -> experimental_reproduction/config_generator.ipynb
                                   |   running_mode: "goal_directed_generation"
                                   v
  STEP 4  ·  OBJECTIVE  Construct the Oracle (scoring function):
                        docking | synthesizability/retrosynthesis | physchem |
                        similarity | structural | xTB/DFT | Boltz
                        -> oracles/ , scoring/
                                   |  
                                   |   
                                   v
  STEP 5  ·  RL LOOP   +==============================================+
                       |  REINFORCEMENT-LEARNING OPTIMIZATION LOOP   O |
                       |                                              |
                       |  a) Sample a batch of molecules (Mamba/RNN)  |
                       |     -> goal_directed_generation/ , models/   |
                       |  b) Score with the Oracle  -> oracles/       |
                       |  c) Diversity filter  -> diversity_filter/   |
                       |  d) MEMORY MANIPULATION (sample-efficiency    |
                       |     core): augmented/experience replay,      |
                       |     hallucinated memory, beam enumeration    |
                       |     -> experience_replay/                    |
                       |        hallucinated_memory/                  |
                       |        beam_enumeration/                     |
                       |  e) Update the agent (policy gradient)       |
                       |     -> goal_directed_generation/             |
                       |          reinforcement_learning.py           |
                       |                                              |
                       |  repeat a-e until the oracle budget is spent |
                       +=====================|========================+
                                             v
  STEP 6  ·  RESULTS    Generated molecules + per-component scores
                        (oracle_history.csv), checkpoints and logs
                        -> experimental_reproduction/part_1/example-output-files/
                                   |
                                   v
  STEP 7  ·  ANALYSE    Visualise synthesis routes / optimization dynamics
                        -> user-specific code
```

**Where each step lives:**

| Step | What happens | Folder |
|------|--------------|--------|
| 1. Setup | Install the `saturn` env (+ docking/retrosynthesis tools if needed) | `setup.sh` |
| 2. Model | Reuse a bundled checkpoint, or pre-train via `distribution_learning` | `experimental_reproduction/checkpoint_models/`, `distribution_learning/`, `models/` |
| 3. Config | Write the configuration JSON | `experimental_reproduction/config_generator.ipynb` |
| 4. Oracle | Construct the scoring function(s) | `oracles/`, `scoring/` |
| 5. RL loop | Sample → score → filter → memory manipulation → update | `goal_directed_generation/`, `experience_replay/`, `hallucinated_memory/`, `beam_enumeration/`, `diversity_filter/` |
| 6. Results | Molecules, scores, checkpoints, logs | logging path in the config (example in `experimental_reproduction/part_1/`) |
| 7. Analysis | Inspect routes & optimization dynamics | `visualization/app.py`, `notebook_analysis/` |

</details>

🧪 Experimental reproduction
-------------------

Reproducing a paper experiment? Start in [`experimental_reproduction/`](experimental_reproduction) — each sub-folder ships prepared configs and its own README. Each component may require additional installation steps (e.g., docking, retrosynthesis, oracles). See the README in each sub-folder for details.

Use the index below to find the sub-folder associated with each paper:

| Paper | Sub-folder(s) |
|-------|---------------|
| Saturn &mdash; main framework [[1]](#-references) | [`main_framework_saturn/`](experimental_reproduction/main_framework) |
| Coupling Saturn with Retrosynthesis Models [[2]](#-references) | [`synthesizability/`](experimental_reproduction/synthesizability) |
| TANGO &mdash; Constrained Synthesizability [[3]](#-references) | [`constrained_synthesizability/`](experimental_reproduction/constrained_synthesizability) |
| Granular Synthesizability Control [[4]](#-references) | [`synthesizability_control/`](experimental_reproduction/synthesizability) |
| Synthesizability with Reaction-Condition Constraints [[5]](#-references) | [`reaction_conditions_synthesizability/`](experimental_reproduction/reaction_conditions_synthesizability) |


Git Hash Code Versions
----------------------
* [Saturn](https://www.nature.com/articles/s42256-026-01200-4): fee0179
* [TANGO Constrained Synthesizability](https://www.nature.com/articles/s43588-026-00959-1): de5cd7f
* [Steerable and Granular Synthesizability Control Pre-print](https://arxiv.org/abs/2505.08774): 468b1f4
* [Reaction-Condition Constrained Synthesizability Pre-print](https://openreview.net/pdf?id=KIab6hg0cT): 4a3e10b


👋 Acknowledgements
----------------
The `Mamba` architecture code was adapted from the following sources:
* [Official Repository](https://github.com/state-spaces/mamba)
* [Mamba Protein Language Model](https://github.com/programmablebio/ptm-mamba)
* [Mamba CPU](https://github.com/kroggen/mamba-cpu)

📜 References
----------
1. [Saturn paper](https://www.nature.com/articles/s42256-026-01200-4)
2. [Generating Synthesizable Molecules - Coupling Saturn with Retrosynthesis Models](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc01476j)
3. [TANGO: Constrained Synthesizability](https://www.nature.com/articles/s43588-026-00959-1)
4. [Granular Synthesizability Control](https://arxiv.org/abs/2505.08774)
5. [Synthesizability with Reaction-Condition Constraints](https://openreview.net/pdf?id=KIab6hg0cT)
6. [Augmented Memory](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)
7. [Beam Enumeration](https://arxiv.org/abs/2309.13957)
8. [GraphGA](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)