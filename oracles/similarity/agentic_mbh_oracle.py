"""
MBH catalyst-scoring oracle for Saturn (the essential MBH delta of this project's
Saturn fork, brought under the Agentic-AI-for-Catalyst-Design roof).

This module is a Saturn ``OracleComponent``: Saturn proposes candidate molecules,
this oracle filters them to valid tertiary-amine MBH catalysts (hard structural
guards to stop the RL agent reward-hacking) and batch-scores the survivors with an
LLM acting as a physical-organic-chemistry judge for the Morita-Baylis-Hillman
(MBH) reaction. DABCO is fixed at 50/100 as the calibration anchor.

It is imported *inside* a Saturn checkout at launch time (see ``launcher.py``),
so it depends on Saturn's ``oracles.oracle_component`` / ``oracles.dataclass``,
which are only importable when Saturn is on ``sys.path``. The LLM call uses the
OpenRouter (OpenAI-compatible) backend to match this repository's ``config.py``.
"""

import os
import json
import time
import concurrent.futures

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Provided by the Saturn checkout at runtime (injected onto sys.path by the launcher).
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters


# --- Hard structural guards (anti reward-hacking) ---------------------------
# Required core: a neutral tertiary amine that is NOT an amide / sulfonamide N.
TERTIARY_AMINE_PATTERN = Chem.MolFromSmarts("[NX3;H0;+0;!$(NC=O);!$(NS=O)]")
# Ban list: any N-H amine or any cationic nitrogen anywhere in the molecule.
FORBIDDEN_NITROGENS = Chem.MolFromSmarts("[NH1,NH2,NH3,n+1,N+1]")

MAX_MOL_WT = 250.0       # DABCO is 112 Da; cap forces compact scaffolds.
MAX_ROTATABLE_BONDS = 3  # MBH transition state punishes floppy chains.


def is_valid_mbh_catalyst(mol) -> bool:
    """Hard physical/structural filters that a candidate must pass before scoring."""
    if mol is None:
        return False
    if not mol.HasSubstructMatch(TERTIARY_AMINE_PATTERN):
        return False
    if mol.HasSubstructMatch(FORBIDDEN_NITROGENS):
        return False
    if Descriptors.MolWt(mol) > MAX_MOL_WT:
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > MAX_ROTATABLE_BONDS:
        return False
    return True


class MBHcatalystscore(OracleComponent):
    """LLM-as-oracle that batch-scores tertiary-amine MBH catalysts (0-100, returned as 0-1).

    Specific parameters (set in the Saturn config JSON under ``specific_parameters``):
        api_key:          OpenRouter API key. Falls back to ``OPENROUTER_API_KEY``.
        model_name:       OpenRouter model id (default: ``google/gemini-3.5-flash``).
        num_calls:        Replicate LLM calls per batch, averaged (default: 3).
        batch_size:       Molecules scored per prompt (default: 16).
        rate_limit_delay: Seconds to sleep between batches (default: 1.0).
    """

    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        sp = self.parameters.specific_parameters
        self.api_key = sp.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MBH oracle needs an OpenRouter key: set 'api_key' in specific_parameters "
                "or export OPENROUTER_API_KEY."
            )

        # Import lazily so the module is importable even where openai isn't installed.
        from openai import OpenAI
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)

        self.model_name = sp.get("model_name", "google/gemini-3.5-flash")
        self.num_calls = int(sp.get("num_calls", 3))
        self.batch_size = int(sp.get("batch_size", 16))
        self.rate_limit_delay = float(sp.get("rate_limit_delay", 1.0))

    def _build_batch_prompt(self, smiles_map: dict) -> str:
        """Heavily contextualized MBH-mechanism prompt over a JSON map of {index: SMILES}."""
        smiles_list_str = json.dumps(smiles_map, indent=2)
        return (
            "You are an expert physical organic chemist evaluating molecules for the discovery of new tertiary amines that can catalyze the Morita-Baylis-Hillman (MBH) reaction.\n"
            "Consider the Morita-Baylis-Hillman (MBH) reaction of methyl acrylate (MA) "
            "with p-nitrobenzaldehyde (pNBA) in methanol.\n\n"

            "### Reaction Context & Mechanism\n"
            "1. Mechanistic Steps:\n"
            "   - Step 1: Nucleophilic attack of the tertiary amine on MA to form a zwitterionic enolate.\n"
            "   - Step 2: Aldol-type C-C bond formation via addition of the enolate to pNBA, forming a zwitterionic alkoxide.\n"
            "   - Step 3: Solvent-mediated proton transfer and subsequent elimination of the amine catalyst to yield the final MBH adduct.\n\n"

            "2. Catalyst Requirements (Tertiary Amines):\n"
            "   - High Nucleophilicity: Essential to efficiently initiate the reaction (Step 1).\n"
            "   - Low Steric Hindrance: Critical to allow attack and to avoid severe steric clashes in the highly congested C-C bond-forming transition state. Compact or bicyclic amines (e.g., DABCO, quinuclidine derivatives) generally outperform bulky acyclic amines (e.g., triethylamine).\n"
            "   - Leaving Group Ability: Must readily eliminate in Step 3 to turn over the catalytic cycle.\n\n"

            "3. Solvent Effects & Rate-Determining Step (RDS) in Methanol:\n"
            "   - Methanol serves as a strong hydrogen-bond donor, significantly stabilizing the highly polar zwitterionic intermediates.\n"
            "   - Because methanol powerfully facilitates proton transfer (Step 3), the Aldol addition (C-C bond formation in Step 2) becomes the strict Rate-Determining Step (RDS).\n\n"

            "### CRITICAL PENALTIES (Apply these ruthlessly):\n"
            "A. Entropic Penalty: MBH catalysts must be highly rigid and compact molecular bullets to organize the transition state. Heavily penalize floppy molecules or long alkyl chains.\n"
            "B. Chemical Realism: Penalize structures with bizarre, unstable, or competing functional groups that look like AI hallucinations rather than synthesizable, stable lab chemicals.\n\n"

            "### Calibration & Benchmarking:\n"
            "To ensure consistency, use these benchmarks to set your scoring scale:\n"
            "- DABCO (1,4-diazabicyclo[2.2.2]octane): 50.0 (The reference catalyst).\n\n"

            "### Task\n"
            "Evaluate the given molecules represented by the following JSON map of index to SMILES:\n"
            f"{smiles_list_str}\n\n"

            "Analyze each given SMILES against the strict requirements for nucleophilicity, sterics, rigidity, and synthetic realism.\n"
            "Predict its expected catalytic performance (factoring in reaction rate and yield) "
            "on a continuous scale from 0.0 (completely inactive/poor) to 100.0 (highly active/excellent).\n\n"

            "### Output Instructions\n"
            "Return ONLY a valid JSON object where keys are the indices provided and values are the floating-point scores. "
            "Do not include markdown formatting (like ```json), reasoning, commentary, or any text outside the JSON object.\n"
            "Example Schema:\n"
            "{\n"
            '  "0": 75.5,\n'
            '  "1": 15.0\n'
            "}"
        )

    def _make_batch_api_call(self, prompt: str, expected_indices: list) -> dict:
        """One LLM call; returns {index: clamped_score_0_100}. Errors score 0.0."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw_scores = json.loads(response.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - never let an API hiccup kill a campaign
            print(f"[MBH oracle] Batch API error: {e}")
            return {int(idx): 0.0 for idx in expected_indices}

        cleaned = {}
        for idx in expected_indices:
            try:
                cleaned[int(idx)] = max(0.0, min(100.0, float(raw_scores.get(str(idx), 0.0))))
            except (TypeError, ValueError):
                cleaned[int(idx)] = 0.0
        return cleaned

    def __call__(self, mols: list) -> np.ndarray:
        """Score a list of RDKit Mols. Returns rewards in [0, 1] (score/100)."""
        results = np.zeros(len(mols), dtype=np.float32)

        all_smiles = []
        valid_indices = []
        for i, mol in enumerate(mols):
            if is_valid_mbh_catalyst(mol):
                all_smiles.append(Chem.MolToSmiles(mol))
                valid_indices.append(i)
            else:
                all_smiles.append(None)  # invalid -> stays 0.0, not sent to the LLM

        for start in range(0, len(valid_indices), self.batch_size):
            batch_indices = valid_indices[start:start + self.batch_size]
            smiles_map = {idx: all_smiles[idx] for idx in batch_indices}
            prompt = self._build_batch_prompt(smiles_map)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_calls) as executor:
                futures = [
                    executor.submit(self._make_batch_api_call, prompt, batch_indices)
                    for _ in range(self.num_calls)
                ]
                batch_runs = [f.result() for f in futures]

            for idx in batch_indices:
                per_call = [run[idx] for run in batch_runs]
                results[idx] = float(sum(per_call) / len(per_call)) / 100.0

            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results
