apt-get update
apt-get install unzip
pip install --upgrade pip

python -m venv venv
source venv/bin/activate
echo "✅ venv activated."

pip install --upgrade pip
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension


python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# add kaggle.json to ~/.kaggle directory
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d ushariranasinghe/car-caption-dataset

unzip car-caption-dataset.zip

apt update
apt install openjdk-8-jdk
update-alternatives --set java /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java
java -version

git clone https://github.com/salaniz/pycocoevalcap.git

cd pycocoevalcap
pip install .

echo "✅ environment setup done."
