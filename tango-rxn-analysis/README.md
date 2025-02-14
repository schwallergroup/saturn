# TANGO with Reaction Constraints Analysis Script

`analyze.py` is a script that outputs various metrics for TANGO-RXN experiments. 

# Reaction Condition Annotation:

This is performed by [Reacon](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc05946h) and requires following the installation [here](https://github.com/schwallergroup/reacon).

# General Metrics
---------------
`# Non-synthesizable`: Number of generated molecules without a solved route by the retrosynthesis model

`# Synthesizable`: Number of generated molecules with a solved route by the retrosynthesis model

`# Synthesizable (with all constraints)`: Number of generated molecules with a solved route by the retrosynthesis model and all constraints satisfied (***building blocks and/or reaction constraints***)

`# Successful Runs`: Number of runs that generated at least one synthesizable molecule (with all constraints)

`Wall Time`: Time taken for each run


Molecule Quality Metrics
------------------------
***Metrics here are reported for generated molecules that are synthesizable (with all constraints)***

`Docking Scores`: QuickVina2-GPU or gnina docking scores

`QED`: Quantitative Estimate of Drug-likeness

`Ligand Efficiency`: Docking score / Number of heavy atoms


Molecular Diversity Metrics
---------------------------
***Metrics here are reported for generated molecules that are synthesizable (with all constraints)***

`# Unique Bemis-Murcko Scaffolds`: Number of unique Bemis-Murcko scaffolds

`IntDiv1`: Internal diversity (1 - intraset Tanimoto distance) - [source](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2020.565644/full)

`#Circles`: Sphere packing diversity metric - [source](https://openreview.net/forum?id=Yo06F8kfMa1)


"Top" Molecule Metrics
----------------------
***Metrics here are reported for generated molecules that are synthesizable (with all constraints) and filtered by the top % (user-specified) of reward***

#### Same Metrics as above:

`Docking Scores`, `QED`, `Ligand Efficiency`, `IntDiv1`, `#Circles`

#### New Metrics:

`Top Graphs Reaction Steps`: Number of reaction steps

`Top Graphs`: PDF of the synthetic routes
