# SATURN: Sample Efficient Generative Molecular Design by Memory Manipulation

Note: contains far fewer capabilities compared to REINVENT 4. The purpose of this repository is a minimal implementation of Augmented Memory, Hallucinated Memory, and Beam Enumeration with a focus only on small molecule generation. 


* track hallucination how many times replace the buffer 
* option to incept SMILES?


Be careful diversity filter stores "min similarity" np.max(bulk tanimoto)

Curriculum learning

Need to be able to:
1. pre-train
2. just sample
3. run RL
4. run CL


Lightning support for pre-train, RL, CL