"""
Adapted from https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/dataset.py.
"""
import torch
from torch.utils.data import Dataset
import numpy as np

from models.generator import Generator
from models.vocabulary import SMILESTokenizer, create_vocabulary

from utils.chemistry_utils import randomize_smiles


class SMILESDataset(Dataset):
    """
    Dataset class for SMILES strings.
    In principle, any string-based representation of molecules can be 
    used, as long as the Tokenizer and Vocabulary are adapted accordingly.
    """
    def __init__(
        self, 
        agent: str,
        dataset_path: str, 
        batch_size: int = 256,
        transfer_learning: bool = False,
        randomize: bool = True,
        max_sequence_length: int = 128
    ):
        self.dataset_path = dataset_path
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.randomize = randomize
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.agent = agent
        self.transfer_learning = transfer_learning
        self.setup()

    def __getitem__(self, idx: int) -> torch.Tensor:
        smiles = self.dataset[idx]
        # Randomize SMILES string if specified 
        # Can enhance chemical space coverage of the model: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0393-0
        smiles = randomize_smiles(smiles) if self.randomize else smiles
        tokens = self.tokenizer.tokenize(smiles)
        encoded = self.vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def read_data_file(self) -> np.ndarray[str]:
        """
        Reads a file with SMILES strings and returns a np.array of the SMILES.
        Removes SMILES exceeded sequence length of 256.
        """
        with open(self.dataset_path, "r") as file:
            return np.array([smiles for smiles in file.read().splitlines() if len(self.tokenizer.tokenize(smiles)) <= self.max_sequence_length])

    def setup(self):
        """
        Performs 3 tasks:
            1. Intializes the Tokenizer
            2. Initializes the Vocabulary
            3. Reads the dataset file and filters based on max_sequence_length

        If transfer_learning is True, the Agent is loaded and the Tokenizer and Vocabulary are extracted.
        """
        if self.transfer_learning:
            # Load model and extract the Tokenizer and Vocabulary
            self.agent = Generator.load_from_file(
                model_path=self.agent,
                device=self.device,
                sampling_mode=False
            )
            self.tokenizer = self.agent.tokenizer
            self.dataset = self.read_data_file()
            self.vocabulary = self.agent.vocabulary
        else:
            self.tokenizer = SMILESTokenizer()
            self.dataset = self.read_data_file()
            self.vocabulary = create_vocabulary(
                smiles=self.dataset,
                tokenizer=self.tokenizer
            )
            # The block of code below was used in the GEAM experiments which pre-trained on ZINC 250k.
            # Training with randomization requires extra tokens to be added to the vocabulary.
            # extra_zinc_tokens = ["[P@H]", "[S@+]"]
            # self.vocabulary.update(extra_zinc_tokens)
        
    @staticmethod
    def collate_fn(encoded_seqs: torch.Tensor) -> torch.Tensor:
        """Converts a list of encoded sequences into a padded tensor."""
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for idx, seq in enumerate(encoded_seqs):
            collated_arr[idx, :seq.size(0)] = seq
        return collated_arr
