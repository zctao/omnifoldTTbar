#!/bin/bash
VENVNAME=${1:-${HOME}/venv_tf2170}

#module purge
module load python/3 scipy-stack
virtualenv --no-download ${VENVNAME}
source ${VENVNAME}/bin/activate

pip install --no-index --upgrade pip

# Install the required packages
pip install --no-index tensorflow==2.17.0
pip install --no-index scikit-learn~=1.5.2
pip install --no-index uproot~=5.3.7
pip install --no-index PyYAML~=6.0
pip install mplhep~=0.3.55
pip install hist~=2.8.0

deactivate