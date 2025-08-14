import os
import time
import subprocess
import tempfile
import shutil
import paramiko
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from oracles.oracle_component import OracleComponent
from oracles.dataclass import OracleComponentParameters



class DFTScore(OracleComponent):
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)
        self.env_name = self.parameters.specific_parameters.get("env_name", None)
        assert self.env_name is not None, "Please provide the Conda environment name with DFTScore installed."

        self.property = self.parameters.specific_parameters.get("property", None)
        assert self.property is not None, "Please provide the DFT property to be calculated."
        self.maximize = self.parameters.specific_parameters.get("maximize", True)
        assert self.maximize in [True, False], "Please provide a valid maximize flag."

        self.time_limit = self.parameters.specific_parameters.get("time_limit", 60)
        assert self.time_limit is not None and self.time_limit >= 30, "The time limit must be >= 30 minutes."

        self.machine = self.parameters.specific_parameters.get("machine", "remote")
        assert self.machine in ["local", "remote"], "Please provide a valid machine name."
        self.hostname = self.parameters.specific_parameters.get("hostname", None)
        self.username = self.parameters.specific_parameters.get("username", None)
        self.queue = self.parameters.specific_parameters.get("queue", "slurm")
        if self.machine == "remote":
            assert self.hostname is not None, "Please provide the hostname of the remote machine."
            assert self.username is not None, "Please provide the username of the remote machine."
            self.remote_path = self.parameters.specific_parameters.get("remote_path", None)
            assert self.remote_path is not None, "Please provide the remote path of the remote machine to save the DFT geometries."

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_geometry_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the directory to save the DFT geometries."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        """
        Execute DFTScore on the SMILES batch.
        """
        # 1. Get SMILES
        smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]

        # 2. Create temporary directories for each SMILES
        # FIXME: Can just generate some hex code folder
        temp_dirs = [os.path.join(self.remote_path, tempfile.mkdtemp().replace("/tmp/", "")) for _ in smiles]

        # 3. Run DFTScore for each SMILES *simultaneously*
        processes = []
        for s, temp_dir in zip(smiles, temp_dirs):
            process = subprocess.Popen([
                "conda",
                "run",
                "-n",
                self.env_name,
                "dft_score",
                "--smiles", str(s),
                "--dir", str(temp_dir),
                "--task", str(self.property),
                "--machine", str(self.machine),
                "--hostname", str(self.hostname),    
                "--username", str(self.username),
                "--time_limit", str(self.time_limit),
                "--queue", str(self.queue)
            ])
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.wait()
        
        start_time = time.time()
        max_wait_time = self.time_limit + 100000  # Time limit with some leeway
        results = []

        # Connect to remote machine with robust error handling
        # Keep re-trying until connection is successful (cluster has limitations on number of concurrent connections)
        ssh = None
        sftp = None
        connected = False
        retry_count = 0
        
        while not connected:
            try:
                # Clean up any existing connection
                if ssh is not None:
                    try:
                        ssh.close()
                    except:
                        pass
                
                ssh = paramiko.SSHClient()
                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh.load_system_host_keys()
                
                # Connect with proper timeouts to avoid SSH banner errors
                ssh.connect(
                    self.hostname, 
                    username=self.username,
                    timeout=30,          # Connection timeout
                    banner_timeout=30,   # SSH banner timeout - key for fixing your error
                    auth_timeout=30      # Authentication timeout
                )
                sftp = ssh.open_sftp()
                connected = True
                
            except paramiko.ssh_exception.SSHException as e:
                retry_count += 1
                wait_time = min(5 * (1.5 ** min(retry_count - 1, 10)), 60)
                time.sleep(wait_time)
                    
            except Exception as e:
                retry_count += 1
                time.sleep(5)
        
        # Wait for all jobs to complete
        try:
            while time.time() - start_time < max_wait_time:
                all_completed = True
                for temp_dir in temp_dirs:
                    try:
                        # Check if result file exists on remote with timeout
                        stdin, stdout, stderr = ssh.exec_command(
                            f"test -f {temp_dir}/{self.property} && echo 'exists' || echo 'missing'",
                            timeout=30
                        )
                        result = stdout.read().decode().strip()
                        if result != "exists":
                            all_completed = False
                            break
                    except Exception as e:
                        print(f"Error checking file status for {temp_dir}: {e}")
                        # If we lose connection during monitoring, try to reconnect
                        try:
                            ssh.close()
                        except:
                            pass
                        
                        # Try to reconnect
                        reconnected = False
                        reconnect_attempts = 0
                        while not reconnected:
                            try:
                                ssh = paramiko.SSHClient()
                                ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                                ssh.load_system_host_keys()
                                ssh.connect(
                                    self.hostname, 
                                    username=self.username,
                                    timeout=30,
                                    banner_timeout=30,
                                    auth_timeout=30
                                )
                                sftp = ssh.open_sftp()
                                reconnected = True
                                print(f"Reconnected to {self.hostname}")
                                
                            except paramiko.ssh_exception.SSHException as reconnect_e:
                                reconnect_attempts += 1
                                print(f"SSH reconnection error on attempt {reconnect_attempts}: {reconnect_e}")
                                wait_time = min(5 * (1.5 ** min(reconnect_attempts - 1, 5)), 30)
                                time.sleep(wait_time)
                                
                            except Exception as reconnect_e:
                                reconnect_attempts += 1
                                print(f"Reconnection error on attempt {reconnect_attempts}: {reconnect_e}")
                                time.sleep(5)
                        
                        all_completed = False
                        break
                
                if all_completed:
                    break

                time.sleep(30)  # Wait for 30 seconds before checking again
                
        except Exception as e:
            print(f"Error during job monitoring: {e}")
            # Continue to try collecting whatever results are available
        
        # Collect results from remote temp_dirs
        for idx, temp_dir in enumerate(temp_dirs):
            try:
                # Read result file directly from remote
                remote_result_path = f"{temp_dir}/{self.property}"
                with sftp.open(remote_result_path, "r") as remote_file:
                    result_value = float(remote_file.read().strip())
                    results.append(result_value)
            except Exception as e:
                print(f"Error reading result for molecule {idx}: {e}")
                # If file doesn't exist or can't be read, assign penalty value
                results.append(-99 if self.maximize else 99)
            
                # Delete the temporary parent directory on remote machine
        try:
            stdin, stdout, stderr = ssh.exec_command(f"rm -rf {self.remote_path}")
            stdout.read()  # Wait for command to complete
        except Exception as e:
            print(f"Warning: Could not delete remote directory {self.remote_path}: {e}")
        
        sftp.close()
        ssh.close()

        # 6. Delete the temporary directories
        if self.machine == "local": 
            for temp_dir in temp_dirs:
                shutil.rmtree(temp_dir)

        return np.array(results, dtype=np.float32)
