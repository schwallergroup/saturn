"""
Adapted from https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/dataset.py.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from models.model import Model
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
        randomize: bool = True
    ):
        self.dataset = self.read_data_file(dataset_path)  # np.ndarray[str]
        self.batch_size = batch_size
        self.randomize = randomize

        # initialize the Tokenizer and Vocabulary based on whether pre-training or fine-tuning is to be performed
        self.agent = agent
        self.transfer_learning = transfer_learning
        self.setup_vocabulary_and_tokenizer()

    def __getitem__(self, idx: int) -> torch.Tensor:
        smiles = self.dataset[idx]
        # randomize SMILES string if specified 
        # can enhance chemical space coverage of the model: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0393-0
        smiles = randomize_smiles(smiles) if self.randomize else smiles
        tokens = self.tokenizer.tokenize(smiles)
        encoded = self.vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)
    
    def read_data_file(self, path: str) -> np.ndarray[str]:
        """Reads a file with SMILES strings and returns a np.array of the SMILES."""
        with open(path, "r") as file:
            return np.array(file.read().splitlines())
        
    def setup_vocabulary_and_tokenizer(self):
        """
        Initializes the Tokenizer and Vocabulary based on whether pre-training or fine-tuning is to be performed.
        """
        if self.transfer_learning:
            # load model
            self.agent = Model.load_from_file(self.agent)
            self.tokenizer = self.agent.tokenizer
            self.vocabulary = self.agent.vocabulary
        else:
            # construct the Tokenizer and Vocabulary from the training data
            self.tokenizer = SMILESTokenizer()
            self.vocabulary = create_vocabulary(
                smiles=self.dataset,
                tokenizer=self.tokenizer
            )
        
    @staticmethod
    def collate_fn(encoded_seqs: torch.Tensor) -> torch.Tensor:
        """Converts a list of encoded sequences into a padded tensor."""
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for idx, seq in enumerate(encoded_seqs):
            collated_arr[idx, :seq.size(0)] = seq
        return collated_arr
