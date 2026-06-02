import numpy as np
import time
import json
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import google.generativeai as genai  # <-- Google's native SDK
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters

# 1. The Required Core
TERTIARY_AMINE_PATTERN = Chem.MolFromSmarts('[NX3;H0;+0;!$(NC=O);!$(NS=O)]')

# 2. The Ban List (Any nitrogen with a hydrogen, or any positively charged nitrogen)
FORBIDDEN_NITROGENS = Chem.MolFromSmarts('[NH1,NH2,NH3,n+1,N+1]')

def is_valid_mbh_catalyst(mol) -> bool:
    """Hard physical and structural filters to prevent RL Agent hacking."""
    if not mol.HasSubstructMatch(TERTIARY_AMINE_PATTERN):
        return False
    if mol.HasSubstructMatch(FORBIDDEN_NITROGENS):
        return False
    if Descriptors.MolWt(mol) > 250:
        return False
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > 3:
        return False
    return True

class MBHcatalystscore(OracleComponent):
    """
    An Oracle Component for the Saturn pipeline that uses the Google Generative AI SDK
    to batch-score tertiary amine catalysts for the MBH reaction.
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        self.api_key = self.parameters.specific_parameters.get("api_key")
        if not self.api_key:
            raise ValueError("Please provide 'api_key' in the specific_parameters.")

        # Configure Google Native Client
        genai.configure(api_key=self.api_key)

        self.model_name = self.parameters.specific_parameters.get("model_name", "gemini-1.5-flash")
        
        # Initialize the Google Gemini Model
        self.model = genai.GenerativeModel(self.model_name)

        self.num_calls = self.parameters.specific_parameters.get("num_calls", 3)
        self.batch_size = self.parameters.specific_parameters.get("batch_size", 16)
        self.rate_limit_delay = self.parameters.specific_parameters.get("rate_limit_delay", 1.0)

    def _build_batch_prompt(self, smiles_map: dict) -> str:
        """Constructs the heavily contextualized batch prompt for the LLM evaluation."""
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
            "- DABCO (1,4-diazabicyclo[2.2.2]octane): 50.0 (The reference catalyst).\n"
            
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
            "  \"0\": 75.5,\n"
            "  \"1\": 15.0\n"
            "}"
        )

    def _make_batch_api_call(self, prompt: str, expected_indices: list) -> dict:
        try:
            # Native Google API Call with enforced JSON mode
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Google SDK parses to `.text`
            content = response.text
            raw_scores = json.loads(content)
            
            cleaned_scores = {}
            for idx in expected_indices:
                score = raw_scores.get(str(idx), 0.0)
                try:
                    cleaned_scores[int(idx)] = max(0.0, min(100.0, float(score)))
                except:
                    cleaned_scores[int(idx)] = 0.0
            return cleaned_scores

        except Exception as e:
            print(f"Batch API Error: {e}")
            return {int(idx): 0.0 for idx in expected_indices}

    def __call__(self, mols: list) -> np.ndarray:
        all_smiles = []
        valid_indices = []
        
        results = np.zeros(len(mols), dtype=np.float32)
        
        for i, mol in enumerate(mols):
            if mol is not None and is_valid_mbh_catalyst(mol):
                all_smiles.append(Chem.MolToSmiles(mol))
                valid_indices.append(i)
            else:
                all_smiles.append(None)

        for i in range(0, len(valid_indices), self.batch_size):
            current_batch_indices = valid_indices[i : i + self.batch_size]
            smiles_map = {idx: all_smiles[idx] for idx in current_batch_indices}
            
            prompt = self._build_batch_prompt(smiles_map)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_calls) as executor:
                futures = [executor.submit(self._make_batch_api_call, prompt, current_batch_indices) 
                           for _ in range(self.num_calls)]
                batch_runs = [f.result() for f in futures]

            for idx in current_batch_indices:
                scores_for_this_mol = [run[idx] for run in batch_runs]
                raw_average = sum(scores_for_this_mol) / len(scores_for_this_mol)
                results[idx] = float(raw_average)  

            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)

        return results