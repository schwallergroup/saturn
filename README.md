# Saturn: Sample-efficient Generative Molecular Design using Memory Manipulation

<img src="saturn.jpeg" alt="Saturn Logo" width="300"/>

`Saturn` is a language model based molecular generative design framework that is focused on **sample-efficient *de novo* small molecule design**. 

In the **experimental_reproduction** sub-folder, prepared files and checkpoint models are provided to reproduce the experiments. 
There is also a `Jupyter` notebook to construct your own configuration files to run `Saturn`.

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
2. [Augmented Memory](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)
3. [Beam Enumeration](https://arxiv.org/abs/2309.13957)
4. [GraphGA](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)
