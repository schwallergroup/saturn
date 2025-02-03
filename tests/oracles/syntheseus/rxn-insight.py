from rxn_insight.reaction import Reaction


rxn_smiles = [
    "BrCCCCBr.O=C1CCc2ccc(O)cc2N1>>O=C1CCc2ccc(OCCCCBr)cc2N1",
    "Clc1cccc(N2CCNCC2)c1Cl.O=C1CCc2ccc(OCCCCBr)cc2N1>>O=C1CCc2ccc(OCCCCN3CCN(c4cccc(Cl)c4Cl)CC3)cc2N1"
]

rxn_smiles = [
    "CCOC(=O)C1CCCNC1.ClCc1ccccc1>>CCOC(=O)C1CCCN(Cc2ccccc2)C1",
    "CCOC(=O)C1CCCN(Cc2ccccc2)C1>>O=C(O)C1CCCN(Cc2ccccc2)C1",
    "NC1CCCCC1.O=C(O)C1CCCN(Cc2ccccc2)C1>>O=C(NC1CCCCC1)C1CCCN(Cc2ccccc2)C1"
]

all_reactions = []
for rxn in rxn_smiles:
    rxn = Reaction(rxn)
    ri = rxn.get_reaction_info()
    all_reactions.append(f"{ri['CLASS']}, {ri['NAME']}")

print(all_reactions)
