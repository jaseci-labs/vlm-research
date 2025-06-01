#!/bin/bash

apt-get update
apt-get install -y openjdk-8-jdk

java -version

mkdir -p /workspace/nlp_tools
cd /workspace/nlp_tools

wget https://dl.fbaipublicfiles.com/stanford-corenlp/stanford-corenlp-4.5.9.zip

unzip stanford-corenlp-4.5.9.zip

rm stanford-corenlp-4.5.9.zip

echo 'export CORENLP_HOME=/workspace/nlp_tools/stanford-corenlp-4.5.9' >> ~/.bashrc
echo 'export PATH=$PATH:$CORENLP_HOME' >> ~/.bashrc

source ~/.bashrc

echo "Java 8 installed and Stanford CoreNLP 4.5.9 downloaded & extracted at /workspace/nlp_tools/stanford-corenlp-4.5.9"
