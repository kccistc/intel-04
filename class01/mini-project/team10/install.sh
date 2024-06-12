# Clone model repositories
git clone https://github.com/GaParmar/img2img-turbo.git
git clone https://huggingface.co/spaces/stabilityai/TripoSR

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip3 install -U pip
pip3 install wheel setuptools
pip3 install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472