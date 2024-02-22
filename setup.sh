conda create -n saturn python=3.10
conda activate saturn
pip install rdkit
conda install morfeus-ml -c conda-forge
conda install "libblas=*=*mkl"
conda install -c conda-forge openbabel
conda install -c conda-forge xtb-python
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c conda-forge tqdm
pip install -U pytest
