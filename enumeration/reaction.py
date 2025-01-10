"""
Code from SynNet to model reactions: https://github.com/wenhao-gao/SynNet
"""
import functools
import json
import gzip
import itertools
from typing import Any, Optional, Set, Tuple, Union, Dict

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from tqdm import tqdm

# the definition of reaction classes below
class Reaction:
    """
    This class models a chemical reaction based on a SMARTS transformation.

    Args:
        template (str): SMARTS string representing a chemical reaction.
        rxnname (str): The name of the reaction for downstream analysis.
        smiles: (str): A reaction SMILES string that macthes the SMARTS pattern.
        reference (str): Reference information for the reaction.
    """

    smirks: str  # SMARTS pattern
    rxn: Chem.rdChemReactions.ChemicalReaction
    num_reactant: int
    num_agent: int
    num_product: int
    reactant_template: Tuple[str, str]
    product_template: str
    agent_template: str
    available_reactants: Tuple[list[str], Optional[list[str]]]
    rxnname: str
    smiles: Any
    reference: Any

    def __init__(self, template=None, rxnname=None, smiles=None, reference=None):

        if template is not None:
            # define a few attributes based on the input
            self.smirks = template.strip()
            self.rxnname = rxnname
            self.smiles = smiles
            self.reference = reference

            # compute a few additional attributes
            self.rxn = self.__init_reaction(self.smirks)

            # Extract number of ...
            self.num_reactant = self.rxn.GetNumReactantTemplates()
            if self.num_reactant not in (1, 2):
                raise ValueError("Reaction is neither uni- nor bi-molecular.")
            self.num_agent = self.rxn.GetNumAgentTemplates()
            self.num_product = self.rxn.GetNumProductTemplates()

            # Extract reactants, agents, products
            reactants, agents, products = self.smirks.split(">")

            if self.num_reactant == 1:
                self.reactant_template = list((reactants,))
            else:
                self.reactant_template = list(reactants.split("."))
            self.product_template = products
            self.agent_template = agents
        else:
            self.smirks = None

    def __init_reaction(self, smirks: str) -> Chem.rdChemReactions.ChemicalReaction:
        """Initializes a reaction by converting the SMARTS-pattern to an `rdkit` object."""
        rxn = AllChem.ReactionFromSmarts(smirks)
        rdChemReactions.ChemicalReaction.Initialize(rxn)
        return rxn

    def load(
        self,
        smirks,
        num_reactant,
        num_agent,
        num_product,
        reactant_template,
        product_template,
        agent_template,
        available_reactants,
        rxnname,
        smiles,
        reference,
    ):
        """
        This function loads a set of elements and reconstructs a `Reaction` object.
        """
        self.smirks = smirks
        self.num_reactant = num_reactant
        self.num_agent = num_agent
        self.num_product = num_product
        self.reactant_template = list(reactant_template)
        self.product_template = product_template
        self.agent_template = agent_template
        self.available_reactants = list(available_reactants)  # TODO: use Tuple[list,list] here
        self.rxnname = rxnname
        self.smiles = smiles
        self.reference = reference
        self.rxn = self.__init_reaction(self.smirks)
        return self

    @functools.lru_cache(maxsize=20)
    def get_mol(self, smi: Union[str, Chem.Mol]) -> Chem.Mol:
        """
        A internal function that returns an `RDKit.Chem.Mol` object.

        Args:
            smi (str or RDKit.Chem.Mol): The query molecule, as either a SMILES
                string or an `RDKit.Chem.Mol` object.

        Returns:
            RDKit.Chem.Mol
        """
        if isinstance(smi, str):
            return Chem.MolFromSmiles(smi)
        elif isinstance(smi, Chem.Mol):
            return smi
        else:
            raise TypeError(f"{type(smi)} not supported, only `str` or `rdkit.Chem.Mol`")

    def visualize(self, name="./reaction1_highlight.o.png"):
        """
        A function that plots the chemical translation into a PNG figure.
        One can use "from IPython.display import Image ; Image(name)" to see it
        in a Python notebook.

        Args:
            name (str): The path to the figure.

        Returns:
            name (str): The path to the figure.
        """
        rxn = AllChem.ReactionFromSmarts(self.smirks)
        d2d = Draw.MolDraw2DCairo(800, 300)
        d2d.DrawReaction(rxn, highlightByReactant=True)
        png = d2d.GetDrawingText()
        open(name, "wb+").write(png)
        del rxn
        return name

    def is_reactant(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is a reactant of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeReactant(smi)

    def is_agent(self, smi: Union[str, Chem.Mol]) -> bool:
        """Checks if `smi` is an agent of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeAgent(smi)

    def is_product(self, smi):
        """Checks if `smi` is a product of this reaction."""
        smi = self.get_mol(smi)
        return self.rxn.IsMoleculeProduct(smi)

    def is_reactant_first(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` is the first reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[0])
        return mol.HasSubstructMatch(pattern)

    def is_reactant_second(self, smi: Union[str, Chem.Mol]) -> bool:
        """Check if `smi` the second reactant in this reaction"""
        mol = self.get_mol(smi)
        pattern = Chem.MolFromSmarts(self.reactant_template[1])
        return mol.HasSubstructMatch(pattern)

    def run_reaction(
        self, reactants: Tuple[Union[str, Chem.Mol, None]], keep_main: bool = True
    ) -> Union[str, None]:
        """Run this reactions with reactants and return corresponding product.

        Args:
            reactants (tuple): Contains SMILES strings for the reactants.
            keep_main (bool): Return main product only or all possibel products. Defaults to True.

        Returns:
            uniqps: SMILES string representing the product or `None` if not reaction possible
        """
        # Input validation.
        if not isinstance(reactants, tuple):
            raise TypeError(f"Unsupported type '{type(reactants)}' for `reactants`.")
        if not len(reactants) in (1, 2):
            raise ValueError(f"Can only run reactions with 1 or 2 reactants, not {len(reactants)}.")

        rxn = self.rxn  # TODO: investigate if this is necessary (if not, delete "delete rxn below")

        # Convert all reactants to `Chem.Mol`
        r: Tuple = tuple(self.get_mol(smiles) for smiles in reactants if smiles is not None)

        if self.num_reactant == 1:
            if len(r) == 2:  # Provided two reactants for unimolecular reaction -> no rxn possible
                return None
            if not self.is_reactant(r[0]):
                return None
        elif self.num_reactant == 2:
            # Match reactant order with reaction template
            if self.is_reactant_first(r[0]) and self.is_reactant_second(r[1]):
                pass
            elif self.is_reactant_first(r[1]) and self.is_reactant_second(r[0]):
                r = tuple(reversed(r))
            else:  # No reaction possible
                return None
        else:
            raise ValueError("This reaction is neither uni- nor bi-molecular.")

        # Run reaction with rdkit magic
        ps = rxn.RunReactants(r)

        # Filter for unique products (less magic)
        # Note: Use chain() to flatten the tuple of tuples
        uniqps = list({Chem.MolToSmiles(p) for p in itertools.chain(*ps)})

        # Sanity check
        if not len(uniqps) >= 1:
            # TODO: Raise (custom) exception?
            raise ValueError("Reaction did not yield any products.")

        del rxn

        if keep_main:
            uniqps = uniqps[:1]
        # >>> TODO: Always return list[str] (currently depends on "keep_main")
        uniqps = uniqps[0]
        # <<< ^ delete this line if resolved.
        return uniqps

    def _filter_reactants(
        self, smiles: list[str], verbose: bool = False
    ) -> Tuple[list[str], list[str]]:
        """
        Filters reactants which do not match the reaction.

        Args:
            smiles: Possible reactants for this reaction.

        Returns:
            :lists of SMILES which match either the first
                reactant, or, if applicable, the second reactant.

        Raises:
            ValueError: If `self` is not a uni- or bi-molecular reaction.
        """
        smiles = tqdm(smiles) if verbose else smiles

        if self.num_reactant == 1:  # uni-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            return (reactants_1,)

        elif self.num_reactant == 2:  # bi-molecular reaction
            reactants_1 = [smi for smi in smiles if self.is_reactant_first(smi)]
            reactants_2 = [smi for smi in smiles if self.is_reactant_second(smi)]

            return (reactants_1, reactants_2)
        else:
            raise ValueError("This reaction is neither uni- nor bi-molecular.")

    def set_available_reactants(self, building_blocks: list[str], verbose: bool = False):
        """
        Finds applicable reactants from a list of building blocks.
        Sets `self.available_reactants`.

        Args:
            building_blocks: Building blocks as SMILES strings.
        """
        self.available_reactants = self._filter_reactants(building_blocks, verbose=verbose)
        return self

    @property
    def get_available_reactants(self) -> Set[str]:
        return {x for reactants in self.available_reactants for x in reactants}

    def asdict(self) -> dict():
        """Returns serializable fields as new dictionary mapping.
        *Excludes* Not-easily-serializable `self.rxn: rdkit.Chem.ChemicalReaction`."""
        import copy

        out = copy.deepcopy(self.__dict__)  # TODO:
        _ = out.pop("rxn")
        return out


class ReactionSet:
    """Represents a collection of reactions, for saving and loading purposes."""

    def __init__(self, rxns: Optional[list[Reaction]] = None):
        self.rxns = rxns if rxns is not None else []

    def load(self, file: str):
        """Load a collection of reactions from a `*.json.gz` file."""
        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"
        with gzip.open(file, "r") as f:
            data = json.loads(f.read().decode("utf-8"))

        for r in data["reactions"]:
            rxn = Reaction().load(
                **r
            )  # TODO: `load()` relies on postional args, hence we cannot load a reaction that has no `available_reactants` for extample (or no template)
            self.rxns.append(rxn)
        return self

    def save(self, file: str) -> None:
        """Save a collection of reactions to a `*.json.gz` file."""

        assert str(file).endswith(".json.gz"), f"Incompatible file extension for file {file}"

        r_list = {"reactions": [r.asdict() for r in self.rxns]}
        with gzip.open(file, "w") as f:
            f.write(json.dumps(r_list).encode("utf-8"))

    def __len__(self):
        return len(self.rxns)

    def _print(self, x=3):
        # For debugging
        for i, r in enumerate(self.rxns):
            if i >= x:
                break
            print(json.dumps(r.asdict(), indent=2))