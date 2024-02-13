"""
# TODO: replace model.py with this file later when the code is ready - will need to re-train all models
Based on the implementation from https://github.com/MolecularAI/reinvent-models
"""
from typing import Tuple, List
import torch
import torch.nn as nn
import numpy as np

# import model architectures
from models.rnn import RNN
from models.decoder_transformer import DecoderTransformer

# import vocabulary
from models.vocabulary import Vocabulary, SMILESTokenizer


class Model:
    """
    Parent class for all models. 

    The network attribute is the SMILES generator model and can be the following architectures:
        1. LSTM RNN
        2. Decoder-only Transformer (based on GPT-2)

    The key methods are:
        1. Sampling SMILES
        2. Calculating the likelihood of generating given SMILES (based on the model's weights)
    """

    def __init__(
        self, 
        vocabulary: Vocabulary, 
        tokenizer: SMILESTokenizer, 
        network_params=None, 
        max_sequence_length: int = 256,
        no_cuda: bool = False
    ):
        """
        Implements an RNN.
        :param vocabulary: Vocabulary to use.
        :param tokenizer: Tokenizer to use.
        :param network_params: Dictionary with all parameters required to correctly initialize the RNN class.
        :param max_sequence_length: The max size of SMILES sequence that can be generated.
        """
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        if not isinstance(network_params, dict):
            network_params = {}

        # TODO: add support for the Decoder-only Transformer
        self.network = RNN(len(self.vocabulary), **network_params)

        if torch.cuda.is_available() and not no_cuda:
            self.network.cuda()
    
        self.nll_loss = nn.NLLLoss(reduction="none")

    def set_mode(self, mode: str):
        if mode == "training":
            self.network.train()
        elif mode == "inference":
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode {mode}")

    @classmethod
    def load_from_file(cls, file_path: str, sampling_mode=False):
        """
        Loads a model from a single file
        :param file_path: input file path
        :return: new instance of the RNN or an exception if it was not possible to load it.
        """
        if torch.cuda.is_available():
            save_dict = torch.load(file_path)
        else:
            save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})
        model = Model(
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            network_params=network_params,
            max_sequence_length=save_dict["max_sequence_length"]
        )
        model.network.load_state_dict(save_dict["network"])
        if sampling_mode:
            model.network.eval()
            
        return model

    def save(self, save_path: str):
        """
        Saves the model to save_path.
        """
        save_dict = {
            "vocabulary": self.vocabulary,
            "tokenizer": self.tokenizer,
            "max_sequence_length": self.max_sequence_length,
            "network": self.network.state_dict(),
            "network_params": self.network.get_params()
        }
        torch.save(save_dict, save_path)

    def likelihood_smiles(self, smiles: np.ndarray[str]) -> torch.Tensor:
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch."""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
            for idx, seq in enumerate(encoded_seqs):
                collated_arr[idx, :seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)

        return self.likelihood(padded_sequences)

    def likelihood(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Retrieves the likelihood of a given sequence.

        :param sequences: (batch_size, sequence_length) A batch of sequences
        :return:  (batch_size) Log likelihood for each example.
        """
        logits, _ = self.network(sequences[:, :-1])  # all steps done at once
        log_probs = logits.log_softmax(dim=2)
        return self.nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample_smiles(
        self, 
        num: int = 128, 
        batch_size: int =128
    ) -> Tuple[List[str], np.ndarray[float]]:
        """
        Samples n SMILES from the model.
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        :return:
            :smiles: (n) A list with SMILES.
            :likelihoods: (n) A list of likelihoods.
        """
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break
            seqs, likelihoods = self._sample(batch_size=size)
            smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del seqs, likelihoods
        return smiles_sampled, np.concatenate(likelihoods_sampled)

    def sample_sequences_and_smiles(self, batch_size=128) -> Tuple[torch.Tensor, List, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]
        return seqs, smiles, likelihoods

    # @torch.no_grad()
    def _sample(self, batch_size=128) -> Tuple[torch.Tensor, torch.Tensor]:
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocabulary["^"]
        input_vector = start_token
        sequences = [self.vocabulary["^"] * torch.ones([batch_size, 1], dtype=torch.long)]
        # NOTE: The first token never gets added in the loop so the sequences are initialized with a start token
        hidden_state = None
        nlls = torch.zeros(batch_size)
        for _ in range(self.max_sequence_length - 1):
            logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)
            input_vector = torch.multinomial(probabilities, 1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            nlls += self.nll_loss(log_probs, input_vector)
            if input_vector.sum() == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, nlls

    def get_network_parameters(self):
        return self.network.parameters()
