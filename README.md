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
1. [Augmented Memory](https://pubs.acs.org/doi/10.1021/jacsau.4c00066)
2. [Beam Enumeration](https://arxiv.org/abs/2309.13957)
3. [GraphGA](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc05372c)