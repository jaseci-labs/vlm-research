apt-get update
apt-get install unzip

python -m venv unsloth_env
source unsloth_env/bin/activate

pip install --upgrade pip  
pip install -r requirements.txt
pip install ipykernel  
pip install kaggle 
pip install openpyxl
pip install datasets, pandas
pip install unsloth
pip install wandb
pip install unsloth>=0.1.7
pip install pillow
pip install numpy
pip install pandas
pip install sentence-transformers>=2.2.2
pip install XlsxWriter
python -m ipykernel install --user --name=unsloth_env --display-name "Python (unsloth_env)"

# Only run Kaggle-related setup if --with-kaggle is passed
if [[ "$1" == "--w-kaggle" ]]; then
    echo "Setting up Kaggle dataset..."

    # Add kaggle.json to ~/.kaggle directory
    mkdir -p ~/.kaggle
    cp kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json

    # Download and unzip dataset
    kaggle datasets download -d ushariranasinghe/car-caption-dataset
    unzip car-caption-dataset.zip
fi
# Install Java 8
apt update
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

pip install wandb
wandb login

