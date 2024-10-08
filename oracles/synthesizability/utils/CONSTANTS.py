# Selected SMARTS from: https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
FUNCTIONAL_GROUPS = {
    # Alkenes
    "Allenic Carbon": "[$([CX2](=C)=C)]",
    "Vinylic Carbon": "[$([CX3]=[CX3])]",
    # Alkynes
    "Acetylenic Carbon": "[$([CX2]#C)]",
    # Carbonyl-containing
    "Ketone": "[CX3H1](=O)[#6]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Carboxylic Acid": "[CX3](=O)[OX2H1]",
    "Acyl Halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Urea (Carbamide)": "[NX3][CX3](=[OX1])[NX3]",
    "Ester and Anhydride": "[#6][CX3](=O)[OX2H0][#6]",
    "Carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
    # Ether
    "Ether": "[OD2]([#6])[#6]",
    "Epoxide": "C1OC1",
    # Amino-groups
    "Primary Amine (and not Amide)": "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
    "Two Primary Amines": "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
    "Secondary Amine and not Amide": "[NX3;H2,H1;!$(NC=O)]",
    "Secondary Amine": "[NX3;H2,H1]",
    "Tertiary Amine and not Amide": "[NX3;H0;!$(NC=O)]",
    "Tertiary Amine": "[NX3;H0]",
    # Imine
    "Substituted or Unsubstituted Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    # Imide
    "Unsubstituted dicarboximide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "Substituted dicarboximide": "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
    "Dicarboximide": "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
    # N-containing
    "Nitrate": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    "Nitrile": "[NX1]#[CX2]",
    "Nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "Nitroso": "[NX2]=[OX1]",
    # O-containing
    "Hydroxyl": "[OX2H]",
    "Hydroxyl in Alcohol": "[#6][OX2H]",
    # S-containing
    "Thiol": "[#16X2H]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    "Carbo-Thiocarboxylate": "[S-][CX3](=S)[#6]",
    "Carbo-Thioester": "S([#6])[CX3](=O)[#6]",
    "Sulfide": "[#16X2H0]",
    "Sulfone": "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",  # Permissive
    "Sulfoxide": "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",  # Permissive
    "Sulfate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
    "Sulfamic Acid": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
    # Halide
    "Halide attached to Carbon": "[#6][F,Cl,Br,I]"
}

# Default TANGO weights for Dense Reward
DEFAULT_TANGO_WEIGHTS = {
    "tanimoto": 0.5,
    "fg": 0.5,
    "fms": 0.5
}
