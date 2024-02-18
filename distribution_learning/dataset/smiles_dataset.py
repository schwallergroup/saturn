"""
Adapted from https://github.com/MolecularAI/reinvent-models/blob/main/reinvent_models/reinvent_core/models/dataset.py.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from models.vocabulary import Vocabulary, SMILESTokenizer


class SMILESDataset(Dataset):
    """
    Dataset class for SMILES strings.
    In principle, any string-based representation of molecules can be 
    used, as long as the Tokenizer and Vocabulary are adapted accordingly.
    """
    def __init__(
        self, 
        training_dataset_path: str, 
        validation_dataset_path: str,
        vocabulary: Vocabulary,
        tokenizer: SMILESTokenizer
    ):
        self.training_dataset = self.read_data_file(training_dataset_path)  # np.ndarray[str]
        self.validation_dataset = self.read_data_file(validation_dataset_path)  # np.ndarray[str]
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> torch.Tensor:
        smiles = self.training_dataset[idx]
        tokens = self.tokenizer.tokenize(smiles)
        encoded = self.vocabulary.encode(tokens)
        return torch.tensor(encoded, dtype=torch.long)  # pylint: disable=E1102

    def __len__(self) -> int:
        return len(self.training_dataset)
    
    def read_data_file(self, path: str) -> np.ndarray[str]:
        """Reads a file with SMILES strings and returns a np.array of the SMILES."""
        with open(path, "r") as file:
            return np.array(file.read().splitlines())
        
    @staticmethod
    def collate_fn(encoded_seqs):
        """Converts a list of encoded sequences into a padded tensor"""
        max_length = max([seq.size(0) for seq in encoded_seqs])
        collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
        for i, seq in enumerate(encoded_seqs):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def calculate_nlls_from_model(model, smiles, batch_size=128):
    """
    Calculates NLL for a set of SMILES strings.
    :param model: Model object.
    :param smiles: List or iterator with all SMILES strings.
    :return : It returns an iterator with every batch.
    """
    dataset = Dataset(smiles, model.vocabulary, model.tokenizer)
    _dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=Dataset.collate_fn)

    def _iterator(dataloader):
        for batch in dataloader:
            nlls = model.likelihood(batch.long())
            yield nlls.data.cpu().numpy()

    return _iterator(_dataloader), len(_dataloader)