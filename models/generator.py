"""
Based on the implementation from https://github.com/MolecularAI/reinvent-models.
"""
from typing import Tuple, List, Union
import torch
import torch.nn as nn
import numpy as np

# Import model architectures
from models.rnn import RNN
from models.decoder import Decoder
from models.mamba import MambaConfig, MambaLMHead

# Import vocabulary
from models.vocabulary import Vocabulary, SMILESTokenizer

from utils.utils import generate_causal_mask


class Generator:
    """
    Parent class for all models. 

    The network attribute is the SMILES generator model and can be the following architectures:
        1. LSTM RNN
        2. Decoder-only Transformer (based on GPT-2)
        3. Mamba

    The key methods are:
        1. Sampling SMILES
        2. Calculating the likelihood of generating given SMILES (based on the model's weights)
        3. Saving and loading the model
    """

    def __init__(
        self, 
        model_architecture: str,
        vocabulary: Vocabulary, 
        tokenizer: SMILESTokenizer, 
        device: str,
        network_params = None, 
        max_sequence_length: int = 128,
    ):
        """
        Initializes the SMILES generative model
        :param model_architecture: Architecture of the SMILES generator
        :param vocabulary: Vocabulary to use
        :param tokenizer: Tokenizer to use
        :param network_params: Dictionary with all parameters required to correctly initialize the specific architecture class
        :param max_sequence_length: The max size of SMILES sequence that can be generated
        """
        assert model_architecture in ["rnn", "decoder", "mamba"], f"Invalid model architecture: {model_architecture}."
        self.model_architecture = model_architecture
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self.network = self._initialize_network(network_params, device)
        self.device = next(self.network.parameters()).device
        self.nll_loss = nn.NLLLoss(reduction="none")

    @classmethod
    def load_from_file(
        cls, 
        model_path: str,
        device: str, 
        sampling_mode=False
    ) -> Union[RNN, Decoder, MambaLMHead]:
        """
        Loads trained model from file.
        :param model_path: Path to the model file 
        :return: SMILES generator model instance
        """
        if torch.cuda.is_available():
            save_dict = torch.load(model_path)
        else:
            save_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

        network_params = save_dict.get("network_params", {})
        model = Generator(
            model_architecture=save_dict["model_architecture"],
            vocabulary=save_dict["vocabulary"],
            tokenizer=save_dict.get("tokenizer", SMILESTokenizer()),
            device=device,
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
            "model_architecture": self.model_architecture,
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
        sequences = [torch.tensor(encode, dtype=torch.long, device=self.device) for encode in encoded]

        def collate_fn(encoded_seqs):
            """Function to take a list of encoded sequences and turn them into a batch."""
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long, device=self.device)  # padded with zeroes
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
        # Full sequences are passed and logits obtained using Teacher Forcing
        if isinstance(self.network, RNN):
            logits, _ = self.network(sequences[:, :-1])

        elif isinstance(self.network, Decoder):
            logits = self.network(sequences[:, :-1])

        elif isinstance(self.network, MambaLMHead):
            causal_output = self.network(sequences[:, :-1])
            logits = causal_output.logits

        log_probs = logits.log_softmax(dim=2)  # (batch_size, sequence_length, vocabulary_size)
        return self.nll_loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)

    def sample_smiles(
        self, 
        num: int = 32, 
        batch_size: int = 32
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

    def sample_sequences_and_smiles(self, batch_size=32) -> Tuple[torch.Tensor, List, torch.Tensor]:
        seqs, likelihoods = self._sample(batch_size=batch_size)
        smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]
        return seqs, smiles, likelihoods

    @torch.no_grad()
    def _sample(self, batch_size=32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples a batch of sequences with code logic for different model architectures:
            1. RNN
            2. Decoder
            3. Mamba
        """
        start_token = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        start_token[:] = self.vocabulary["^"]
        input_vector = start_token
        sequences = [self.vocabulary["^"] * torch.ones([batch_size, 1], dtype=torch.long, device=self.device)]

        hidden_state = None  # For RNN
        # Generate full Causal Mask and slice it during Autoregressive generation
        causal_mask = generate_causal_mask(self.max_sequence_length, device=self.device)  # For Decoder
        
        nlls = torch.zeros(batch_size, device=self.device)  # Track Negative Log-Likelihoods

        # Autoregressive generation
        for idx in range(1, self.max_sequence_length + 1, 1):

            if isinstance(self.network, RNN):
                # RNN requires hidden state
                logits, hidden_state = self.network(input_vector.unsqueeze(1), hidden_state)
                logits = logits.squeeze(1)

            elif isinstance(self.network, Decoder):
                input_vector = torch.cat(sequences, 1)
                # Slice the Causal Mask to the current sequence length
                logits = self.network(input_vector, causal_mask[:idx, :idx])
                # Extract logits at the last position
                logits = logits[:, -1, :].squeeze(1)  # (batch_size, vocabulary_size)

            elif isinstance(self.network, MambaLMHead):
                input_vector = torch.cat(sequences, 1)
                causal_output = self.network(input_vector, num_last_tokens=1)
                # Extract logits at the last position
                logits = causal_output.logits[:, -1, :]  # (batch_size, vocabulary_size)
                hidden_state = causal_output.hidden_states  # TODO: Unused?

            probabilities = logits.softmax(dim=1)
            log_probs = logits.log_softmax(dim=1)  # For Negative Log-Likelihood calculation
            input_vector = torch.multinomial(probabilities, num_samples=1).view(-1)
            sequences.append(input_vector.view(-1, 1))
            nlls += self.nll_loss(log_probs, input_vector)

            # Stop sampling if all sequences have generated the stop token
            if input_vector.sum() == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, nlls

    def get_network_parameters(self):
        return self.network.parameters()
    
    def get_num_params(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def _initialize_network(
        self, 
        network_params: Union[dict, None],
        device: str
    ) -> Union[RNN, Decoder, MambaLMHead]:
        """
        Initializes the network based on the model type.
        """
        if not isinstance(network_params, dict):
            network_params = {}

        if self.model_architecture == "rnn":
            network = RNN(len(self.vocabulary), **network_params)

        elif self.model_architecture == "decoder":
            network = Decoder(len(self.vocabulary), **network_params)

        elif self.model_architecture == "mamba":
            if network_params.get("config") is None:
                network = MambaLMHead(MambaConfig(len(self.vocabulary)), device=device)
            else:
                network = MambaLMHead(network_params["config"], device=device)

        network.to(device)

        return network
