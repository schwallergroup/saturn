"""
Adapted from GEAM: https://openreview.net/forum?id=sLGliHckR8
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/docking.py
https://anonymous.4open.science/r/GEAM-45EF/utils_sac/utils.py

The implementation below is based on the above code-base for fair and exact comparison with GEAM.
"""
from typing import Tuple
import os
import sys
from shutil import rmtree
import subprocess
from multiprocessing import Manager
from multiprocessing import Process
from multiprocessing import Queue
import numpy as np
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import Mol, QED
from openbabel import pybel

# SA Score
from oracles.synthesizability.sascorer import calculateScore


class DockingVina(object):
    def __init__(self, target):
        super().__init__()
        
        if target == 'fa7':
            self.box_center = (10.131, 41.879, 32.097)
            self.box_size = (20.673, 20.198, 21.362)
        elif target == 'parp1':
            self.box_center = (26.413, 11.282, 27.238)
            self.box_size = (18.521, 17.479, 19.995)
        elif target == '5ht1b':
            self.box_center = (-26.602, 5.277, 17.898)
            self.box_size = (22.5, 22.5, 22.5)
        elif target == 'jak2':
            self.box_center = (114.758,65.496,11.345)
            self.box_size= (19.033,17.929,20.283)
        elif target == 'braf':
            self.box_center = (84.194,6.949,-7.081)
            self.box_size = (22.032,19.211,14.106)
        self.vina_program = '/home/<your path>/Desktop/saturn/oracles/docking/docking_grids/qvina02'
        self.receptor_file = f'/home/<your path>/Desktop/saturn/oracles/docking/docking_grids/{target}.pdbqt'
        self.exhaustiveness = 1
        self.num_sub_proc = 10
        self.num_cpu_dock = 5
        self.num_modes = 10
        self.timeout_gen3d = 30
        self.timeout_dock = 100

        i = 0
        while True:
            tmp_dir = f'tmp/tmp{i}'
            if not os.path.exists(tmp_dir):
                print(f'Docking tmp dir: {tmp_dir}')
                os.makedirs(tmp_dir)
                self.temp_dir = tmp_dir
                break
            i += 1

    def gen_3d(self, smi, ligand_mol_file):
        """
            generate initial 3d conformation from SMILES
            input :
                SMILES string
                ligand_mol_file (output file)
        """
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        """
            run_docking program using subprocess
            input :
                receptor_file
                ligand_mol_file
                ligand_pdbqt_file
                docking_pdbqt_file
            output :
                affinity list for a input molecule
        """
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' %(self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' %(self.box_size)
        run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d' % (self.num_modes)
        run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        """
            put data to queue
            input: queue
                data = [(idx1,smi1), (idx2,smi2), ...]
                num_sub_proc (for end signal)
        """
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        """
            generate subprocess for docking
            input
                q (queue)
                return_dict
                sub_id: subprocess index for temp file
        """
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print(e)
                print("gen_3d unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print(e)
                print("docking unexpected error:", sys.exc_info())
                print("smiles: ", smi)
                return_dict[idx] = 99.9
                continue
            if len(affinity_list)==0:
                affinity_list.append(99.9)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity

    def predict(self, smiles_list):
        """
            input SMILES list
            output affinity list corresponding to the SMILES list
            if docking is fail, docking score is 99.9
        """
        data = list(enumerate(smiles_list))
        q1 = Queue()
        manager = Manager()
        return_dict = manager.dict()
        proc_master = Process(target=self.creator,
                              args=(q1, data, self.num_sub_proc))
        proc_master.start()

        procs = []
        for sub_id in range(0, self.num_sub_proc):
            proc = Process(target=self.docking_subprocess,
                           args=(q1, return_dict, sub_id))
            procs.append(proc)
            proc.start()

        q1.close()
        q1.join_thread()
        proc_master.join()
        for proc in procs:
            proc.join()
        keys = sorted(return_dict.keys())
        affinity_list = list()
        for key in keys:
            affinity = return_dict[key]
            affinity_list += [affinity]
        return affinity_list
    
    def __del__(self):
        if os.path.exists(self.temp_dir):
            rmtree(self.temp_dir)
            print(f'{self.temp_dir} removed')


def reward_vina(
    smis: np.ndarray[str], 
    predictor: DockingVina
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    raw_docking_scores = - np.array(predictor.predict(smis))
    rewards = np.clip(raw_docking_scores, 0, None)
    return raw_docking_scores, rewards


def reward_qed(
    mols: np.ndarray[Mol]
) -> np.ndarray[float]:
    return np.array([QED.qed(m) for m in mols])


def reward_sa(
    mols: np.ndarray[Mol]
) -> Tuple[np.ndarray[float], np.ndarray[float]]:
    raw_sa = np.array([calculateScore(m) for m in mols])
    sa_rewards = np.array([(10 - raw_score) / 9 for raw_score in raw_sa])
    return raw_sa, sa_rewards


class GEAMOracle(OracleComponent):
    """
    GEAM's Oracle that combines 3 individual Oracle Components:
        1. QuickVina 2 Docking
        2. QED
        3. SA Score
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.vina_oracle = DockingVina(parameters.specific_parameters["target"])
        
    def __call__(self, mols: np.ndarray[Mol]) -> np.ndarray[float]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        return self._compute_property(smiles, mols)
    
    def _compute_property(
        self, 
        smiles: np.ndarray[str], 
        mols: np.ndarray[Mol],
    ) -> Tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """
        Run GEAM's Oracle and return the aggregated reward.
        """
        raw_vina, vina_rewards = reward_vina(smiles, self.vina_oracle)
        qed_rewards = reward_qed(mols)
        raw_sa, sa_rewards = reward_sa(mols)
        # Formula used in GEAM paper
        aggregated_rewards = (np.clip(vina_rewards, 0, 20) / 20) * (qed_rewards) * (sa_rewards)
        # Failed Vina scores are -99.9, multiple these by -1 to make them 99.9 for easier parsing later
        raw_vina[raw_vina == -99.9] = 99.9
        return (raw_vina, qed_rewards, raw_sa, aggregated_rewards)
