apt-get update
pip install --upgrade pip

python -m venv venv
source venv/bin/activate
echo "✅ venv activated."

pip install --upgrade pip
pip install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension

python -m ipykernel install --user --name=venv --display-name "Python (venv)"
