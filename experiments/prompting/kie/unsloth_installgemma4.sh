#!/bin/bash
set -e  

apt update
apt install -y git unzip wget

# ---------------------------------------------------------------------------
# Python venv
# ---------------------------------------------------------------------------
python -m venv unsloth_env
source unsloth_env/bin/activate
pip install --upgrade pip

# ---------------------------------------------------------------------------
# Core Python deps
# ---------------------------------------------------------------------------
pip install ipykernel==6.30.1
pip install kaggle
pip install openpyxl
pip install datasets pandas
pip install wandb
pip install pillow
pip install numpy
pip install "sentence-transformers>=2.2.2"
pip install XlsxWriter
pip install huggingface-hub
pip install PyYAML python-Levenshtein

# ---------------------------------------------------------------------------
# Unsloth (must come before the transformers unpin below)
# ---------------------------------------------------------------------------
pip install unsloth

# ---------------------------------------------------------------------------
# unsloth-zoo pins transformers<=5.5.0, but gemma4_unified needs >=5.10.0.
# Install the newer transformers WITHOUT letting pip resolve unsloth-zoo's
# dep constraints (which would downgrade it right back).
# ---------------------------------------------------------------------------
pip install --no-deps --upgrade "transformers>=5.10.0" \
  || pip install --no-deps --upgrade "git+https://github.com/huggingface/transformers.git"

# Sanity check: confirm transformers recognizes gemma4_unified before proceeding
python -c "from transformers import CONFIG_MAPPING; assert 'gemma4_unified' in CONFIG_MAPPING, 'gemma4_unified not registered in transformers'; print('transformers OK: gemma4_unified registered')"

# ---------------------------------------------------------------------------
# Register the venv as a Jupyter kernel
# ---------------------------------------------------------------------------
python -m ipykernel install --user --name=unsloth_env --display-name "Python (unsloth_env)"

# ---------------------------------------------------------------------------
# Java 8 (required by Stanford CoreNLP / SPICE)
# ---------------------------------------------------------------------------
apt install -y openjdk-8-jdk
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
java -version

# ---------------------------------------------------------------------------
# pycocoevalcap (BLEU / METEOR / CIDEr / SPICE metrics)
# ---------------------------------------------------------------------------
git clone https://github.com/salaniz/pycocoevalcap.git
cd pycocoevalcap
pip install .
cd ..

# ---------------------------------------------------------------------------
# Stanford CoreNLP 3.6.0 (specifically required by SPICE inside pycocoevalcap)
# Leaving this in because SPICE pins to this version.
# ---------------------------------------------------------------------------
mkdir -p spice-1.0
cd spice-1.0
wget http://nlp.stanford.edu/software/stanford-corenlp-3.6.0.zip
unzip stanford-corenlp-3.6.0.zip
cd ..

# ---------------------------------------------------------------------------
# Stanford CoreNLP 4.5.10 (general-purpose NLP, separate from SPICE)
# ---------------------------------------------------------------------------
mkdir -p /workspace/nlp_tools
cd /workspace/nlp_tools

echo "Downloading Stanford CoreNLP 4.5.10..."
wget --https-only --no-check-certificate https://nlp.stanford.edu/software/stanford-corenlp-4.5.10.zip
unzip stanford-corenlp-4.5.10.zip
rm stanford-corenlp-4.5.10.zip

echo 'export CORENLP_HOME=/workspace/nlp_tools/stanford-corenlp-4.5.10' >> ~/.bashrc
echo 'export PATH=$PATH:$CORENLP_HOME' >> ~/.bashrc
export CORENLP_HOME=/workspace/nlp_tools/stanford-corenlp-4.5.10
export PATH=$PATH:$CORENLP_HOME

echo "Java 8 and Stanford CoreNLP 4.5.10 setup complete at $CORENLP_HOME"

# ---------------------------------------------------------------------------
# wandb login — set WANDB_API_KEY env var BEFORE running this script, or
# remove this line and log in manually afterward. Otherwise it will hang.
# ---------------------------------------------------------------------------
if [ -n "$WANDB_API_KEY" ]; then
    wandb login --relogin "$WANDB_API_KEY"
else
    echo "WANDB_API_KEY not set; skipping wandb login. Run 'wandb login' manually later."
fi

echo "Install complete."