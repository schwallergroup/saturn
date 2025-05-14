conda create -n saturn python=3.10
conda activate saturn
pip install rdkit==2023.9.5
conda install morfeus-ml -c conda-forge
conda install -c conda-forge openbabel
conda install -c conda-forge xtb-python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install blas=1.0=mkl
conda install -c conda-forge mkl=2021.4.0=h8d4b97c_729
conda install -c conda-forge tqdm
pip install pandas==2.2.1
pip install numpy==1.26.4
pip install einops==0.7.0
pip install -U pytest==8.1.1
pip install scipy==1.10.0
pip install pathos
pip install causal-conv1d==1.2.0.post2
pip install mamba-ssm==1.2.0.post1
