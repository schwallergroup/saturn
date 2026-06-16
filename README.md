# Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation

<img src="saturn.jpeg" alt="Saturn Logo" width="300"/>

`Saturn` is a language model based molecular generative design framework that is focused on **sample-efficient *de novo* small molecule design**. 

In the **experimental_reproduction** sub-folder, prepared files and checkpoint models are provided to reproduce the experiments. 
There is also a `Jupyter` notebook to construct your own configuration files to run `Saturn`.

---

## Adaptation: *De Novo* Discovery of Morita–Baylis–Hillman (MBH) Catalysts

This fork adapts `Saturn` to a chemistry-specific design campaign: the **generative discovery of
new tertiary-amine catalysts for the Morita–Baylis–Hillman (MBH) reaction** (methyl acrylate +
*p*-nitrobenzaldehyde in methanol). Instead of optimising toward a numerical property predictor,
the generative agent is steered by an **LLM-based oracle** that scores each proposed molecule for
its expected catalytic performance.

### What was added

* **A new oracle component — `MBH_catalyst_score`**
  ([`oracles/similarity/MBH_catalyst_score.py`](oracles/similarity/MBH_catalyst_score.py)),
  registered in [`oracles/utils.py`](oracles/utils.py). It:
  * Applies **hard structural filters** to prevent reward hacking before any scoring
    (`is_valid_mbh_catalyst`): the molecule must contain a tertiary amine
    (`[NX3;H0;+0]`), must not contain N–H or positively charged nitrogen, must have
    MW ≤ 250, and ≤ 3 rotatable bonds (compact, rigid scaffolds).
  * Sends the surviving molecules to **Google Gemini** (`google-generativeai` SDK) using a
    heavily contextualised prompt that encodes the MBH mechanism, catalyst requirements
    (nucleophilicity, low steric hindrance, leaving-group ability), methanol solvent effects,
    entropic/realism penalties, and a **DABCO = 50.0 calibration anchor**.
  * **Batches** molecules (`batch_size`) and averages **multiple independent LLM calls per batch**
    (`num_calls`) to reduce scoring variance, returning a continuous score in `[0, 100]`.

* **A campaign configuration** — [`MBH/MBH_trial.json`](MBH/MBH_trial.json) — a
  `goal_directed_generation` run combining `MBH_catalyst_score` with a `mw` component
  (double-sigmoid, target ~50–400) under a 1500-evaluation oracle budget, using the
  `mamba` architecture and a ZINC-250k prior.

* **A completed run** — [`MBH/mbh_catalyst_run/`](MBH/mbh_catalyst_run/) — containing the model
  checkpoints, logs, and the oracle history (`oracle_history_MBH_17.csv`).

* **A publication-quality plotting script** —
  [`MBH/mbh_catalyst_run/plot_score_vs_calls.py`](MBH/mbh_catalyst_run/plot_score_vs_calls.py) —
  which plots the MBH catalyst score against cumulative oracle calls (individual evaluations,
  moving average, cumulative maximum) with the aggregated reward on a secondary axis, exporting
  both `.png` and `.svg`.

### Running the MBH campaign

1. Edit [`MBH/MBH_trial.json`](MBH/MBH_trial.json) and set your Gemini `api_key`, `model_name`,
   and the logging/checkpoint paths.
2. Launch the campaign from the repository root:

        $ python saturn.py MBH/MBH_trial.json

3. Generate the score-vs-calls figure (requires `pandas` + `matplotlib`):

        $ python MBH/mbh_catalyst_run/plot_score_vs_calls.py

> **Note:** `MBH_catalyst_score` requires the `google-generativeai` package and a valid API key.
> Do **not** commit real API keys — supply them via the config file locally or an environment
> variable and keep them out of version control.

---

Git Hash Code Versions
----------------------
* [Saturn Pre-print](https://arxiv.org/abs/2405.17066): fee0179
* [TANGO Constrained Synthesizability Pre-print](https://arxiv.org/abs/2410.11527): de5cd7f
* [Steerable and Granular Synthesizability Control Pre-print](https://arxiv.org/abs/2505.08774): 468b1f4

Installation
-------------

1. Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
2. Clone this Git repository
3. Open terminal and install the `saturn` environment:
   
        $ source setup.sh

Potential Installation Issues
-----------------------------
* `GLIBCXX_3.4.29` version not found - thank you to [@PatWalters](https://github.com/PatWalters) for flagging this and solving via:

        $ conda uninstall openbabel 
        $ conda install gcc_linux-64
        $ conda install gxx_linux-64
        $ conda install -c conda-forge openbabel

* `causal-conv1d` and `mamba-ssm` installation error - see [Issue 1](https://github.com/schwallergroup/saturn/issues/1) - thank you to [@surendraphd](https://github.com/surendraphd) for sharing their solution.

System Requirements
-------------------

* Python 3.10
* Cuda-enabled GPU (CPU-only works but runs times will be much slower)
* Tested on Linux


Acknowledgements
----------------
The `Mamba` architecture code was adapted from the following sources:
* [Official Repository](https://github.com/state-spaces/mamba)
* [Mamba Protein Language Model](https://github.com/programmablebio/ptm-mamba)
* [Mamba CPU](https://github.com/kroggen/mamba-cpu)

References
----------
1. [Saturn Pre-print](https://arxiv.org/abs/2405.17066)
2. [Generating Synthesizable Molecules - Coupling Saturn with Retrosynthesis Models](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d5sc01476j)
3. [TANGO Constrained Synthesizability Pre-print](https://arxiv.org/abs/2410.11527)
4. [Steerable and Granular Synthesizability Control Pre-print](https://arxiv.org/abs/2505.08774)
5. [Augmented Memory](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)
6. [Beam Enumeration](https://arxiv.org/abs/2309.13957)
7. [GraphGA](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)
