import numpy as np

from models.generator import Generator

def get_indices_of_unique_smiles(smiles: np.ndarray[str]) -> np.ndarray[int]:
    """
    Get the indices of unique SMILES in a sampled batch.
    """
    _, indices = np.unique(smiles, return_index=True)
    sorted_indices = np.sort(indices)
    return sorted_indices

# TODO: can move this to the class itself
def sample_unique_sequences(
    agent: Generator, 
    batch_size: int
):
    seqs, smiles, agent_likelihood = agent.sample_sequences_and_smiles(batch_size)
    unique_indices = get_indices_of_unique_smiles(smiles)

    # Get unique sequences, smiles, and agent likelihoods
    seqs = seqs[unique_indices]
    smiles = np.array(smiles)
    smiles = smiles[unique_indices]
    agent_likelihood = agent_likelihood[unique_indices]

    return seqs, smiles, agent_likelihood
