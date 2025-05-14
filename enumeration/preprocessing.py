"""
Code from SynNet to handle Building Blocks and reactions.
"""
from pathlib import Path

from tqdm import tqdm
from enumeration.reaction import Reaction
import multiprocessing 
import json
import os
from typing import Optional, List

# For multiprocessing
MAX_PROCESSES = min(32, multiprocessing.cpu_count()) - 1

from enumeration.reaction import ReactionSet
from enumeration.utils import building_block_passes_property_filter


class BuildingBlockFilter:
    """Filter building blocks."""

    building_blocks: list[str]
    building_blocks_filtered: list[str]
    rxn_templates: list[str]
    rxns: list[Reaction]
    rxns_initialised: bool

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        processes: int = MAX_PROCESSES,
        verbose: bool = False
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates

        # Init reactions
        self.rxns = [Reaction(template=template) for template in self.rxn_templates]
        
        # Init other stuff
        self.processes = processes
        self.verbose = verbose
        self.rxns_initialised = False

    def _match_mp(self):
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(bblocks: list[str], _rxn: Reaction):
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            self.rxns = pool.map(func, self.rxns)
        return self

    def _init_rxns_with_reactants(self):
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        self.rxns = tqdm(self.rxns) if self.verbose else self.rxns
        if self.processes == 1:
            self.rxns = [rxn.set_available_reactants(self.building_blocks) for rxn in self.rxns]
        else:
            self._match_mp()

        self.rxns_initialised = True
        return self

    def filter(self):
        """Filters out building blocks which do not match a reaction template."""
        if not self.rxns_initialised:
            self = self._init_rxns_with_reactants()
        matched_bblocks = {x for rxn in self.rxns for x in rxn.get_available_reactants}
        self.building_blocks_filtered = list(matched_bblocks)
        return self


class BuildingBlockFileHandler:
    def _load_csv(self, file: str) -> list[str]:
        """Load building blocks as smiles from `*.csv` or `*.csv.gz`."""
        import pandas as pd

        return pd.read_csv(file)["SMILES"].to_list()
    
    def _load_smi(self, file: str) -> list[str]:
        """Load building blocks from .smi"""
        import pandas as pd

        return pd.read_csv(file, header=None, names=["SMILES"])

    def load(self, 
             file: str,
             property_filter: bool = True) -> list[str]:
        """Load building blocks from file."""
        file = Path(file)
        if ".csv" in file.suffixes:
            smiles = self._load_csv(file)
        
        elif ".smi" in file.suffixes:
            smiles = self._load_smi(file)
        else:
            raise NotImplementedError
        
        # If property filter, use pre-defined filter
        if property_filter:
            
            from pathos import multiprocessing as mp
            with mp.Pool(processes=MAX_PROCESSES) as pool:
                smiles["passes_filter"] = pool.map(building_block_passes_property_filter, smiles["SMILES"].values)
                smiles = smiles[smiles["passes_filter"] == True]["SMILES"].values
        
        return smiles

    def _save_csv(self, file: Path, building_blocks: list[str]):
        """Save building blocks to `*.csv.gz`"""
        import pandas as pd

        # remove possible 1 or more extensions, i.e.
        # <stem>.csv OR <stem>.csv.gz --> <stem>
        file_no_ext = file.parent / file.stem.split(".")[0]
        file = (file_no_ext).with_suffix(".csv.gz")
        # Save
        df = pd.DataFrame({"SMILES": building_blocks})
        df.to_csv(file, compression="gzip")
        return None

    def save(self, file: str, building_blocks: list[str]):
        """Save building blocks to file."""
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        if ".csv" in file.suffixes:
            self._save_csv(file, building_blocks)
        else:
            raise NotImplementedError


class ReactionTemplateFileHandler:
    def load(self, 
             file: str, 
             names: Optional[list] = None) -> list[str]:
        """Load reaction templates from file."""

        # additional modification for our .json format
        if file.endswith(".json"):
            with open(file, "r") as f:
                data = [json.loads(line) for line in f]
                rxn_templates = [rxn["smirks"] for rxn in data if any(name in rxn["name"].lower() for name in names)]
                
                # Filter reactions that have more than 2 reactants
                rxn_templates = [template for template in rxn_templates if (len(template.split(">>")[0].split(".")) <= 2)]

        # else normal pipeline
        else:
            with open(file, "rt") as f:
                rxn_templates = f.readlines()

            rxn_templates = [tmplt.strip() for tmplt in rxn_templates]

        if not all([self._validate(t)] for t in rxn_templates):
            raise ValueError("Not all reaction templates are valid.")

        return rxn_templates

    def _validate(self, rxn_template: str) -> bool:
        """Validate reaction templates.

        Checks if:
          - reaction is uni- or bimolecular
          - has only a single product

        Note:
          - only uses std-lib functions, very basic validation only
        """
        reactants, agents, products = rxn_template.split(">")
        is_uni_or_bimolecular = len(reactants) == 1 or len(reactants) == 2
        has_single_product = len(products) == 1

        return is_uni_or_bimolecular and has_single_product


def match_bbs(
    bbs_file: str,
    rxn_templates_file: str,
    save_folder: str,
    file_name: str,
    rxn_list: List[str] = None
) -> None:
    """
    Execute first step (building blocks matching with our templates and building blocks file)
    """
    # Load assets
    bblocks = BuildingBlockFileHandler().load(bbs_file)
    rxn_templates = ReactionTemplateFileHandler().load(file=rxn_templates_file,
                                                       names=rxn_list)

    bbf = BuildingBlockFilter(
        building_blocks=bblocks,
        rxn_templates=rxn_templates,
        verbose=False,
        processes=MAX_PROCESSES,
    )
    # Time intensive task...
    bbf.filter()

    # ... and save to disk
    bblocks_filtered = bbf.building_blocks_filtered

    # Filter if reactions do not have available_bbs
    good_rxns = [rxn for rxn in bbf.rxns if not any(len(bb)==0 for bb in rxn.available_reactants)]

    # Save collection of reactions which have "available reactants" set (for convenience)
    rxn_collection = ReactionSet(good_rxns)
    output_file_name = os.path.join(save_folder, file_name)
    rxn_collection.save(output_file_name)

    print(f"Total number of building blocks {len(bblocks):d}")
    print(f"Matched number of building blocks {len(bblocks_filtered):d}")
    print(
        f"{len(bblocks_filtered)/len(bblocks):.2%} of building blocks applicable for the reaction templates."
    )

    print("Completed.")
