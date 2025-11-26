#!/bin/bash
apt update
apt install -y git unzip

python -m venv unsloth_env
source unsloth_env/bin/activate

pip install --upgrade pip  
pip install ipykernel  
pip install kaggle 
pip install openpyxl
pip install datasets pandas
pip install unsloth
pip install wandb
pip install unsloth>=0.1.7
pip install pillow
pip install numpy
pip install pandas
pip install sentence-transformers>=2.2.2
pip install XlsxWriter
pip install openpyxl
python -m ipykernel install --user --name=unsloth_env --display-name "Python (unsloth_env)"


# Install Java 8
#apt update
apt install openjdk-8-jdk
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
java -version

git clone https://github.com/salaniz/pycocoevalcap.git

# Move into the directory
cd pycocoevalcap

# Install with pip
pip install .

mkdir -p spice-1.0
cd spice-1.0
wget http://nlp.stanford.edu/software/stanford-corenlp-3.6.0.zip
unzip stanford-corenlp-3.6.0.zip

# Install Java 8 and unzip
apt-get update
apt-get install -y openjdk-8-jdk unzip wget

# Check Java version
java -version

# Prepare directory
mkdir -p /workspace/nlp_tools
cd /workspace/nlp_tools

# Download CoreNLP with error handling
echo "Downloading Stanford CoreNLP 4.5.10..."
wget --https-only --no-check-certificate https://nlp.stanford.edu/software/stanford-corenlp-4.5.10.zip

# Unzip if successful
unzip stanford-corenlp-4.5.10.zip
rm stanford-corenlp-4.5.10.zip

# Add environment variables (optional but recommended for reuse in terminal)
echo 'export CORENLP_HOME=/workspace/nlp_tools/stanford-corenlp-4.5.10' >> ~/.bashrc
echo 'export PATH=$PATH:$CORENLP_HOME' >> ~/.bashrc

# Export for current shell session
export CORENLP_HOME=/workspace/nlp_tools/stanford-corenlp-4.5.10
export PATH=$PATH:$CORENLP_HOME

echo "Java 8 and Stanford CoreNLP 4.5.10 setup complete at $CORENLP_HOME"

# Install Python packages
pip install wandb numpy pycocoevalcap sentence-transformers
wandb login

