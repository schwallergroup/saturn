# saturn
Sample Efficient Generative Molecular Design using Memory Manipulation

Note: contains far fewer capabilities compared to REINVENT 4. The purpose of this repository is a minimal implementation of Augmented Memory, Hallucinated Memory, and Beam Enumeration with a focus only on small molecule generation. 


* oracle budget 2,000
* track hallucination how many times replace the buffer
* batch size
* seed control
* oracle caching
* oracle flag to allow oracle repeats
* canonical smiles storing
* every generated smiles needs to be canonicalized 
* option to incept SMILES????


Be careful diversity filter stores "min similarity" np.max(bulk tanimoto)

Curriculum learning