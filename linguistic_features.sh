#!/bin/bash
# script name: linguistic_features.sh
#SBATCH --job-name="Linguistic feature extraction"
#SBATCH --comment="Extract linguistic features from the dataset"
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-02:00:00
#SBATCH --output=outs/lingist.out

# export ENV_MODE="permanent"
# export ENV_NAME="linguistic-features"

# module load python/3.11.11

pip install pandas
pip install spacy
pip install numpy
pip install textstat
pip install lexical-diversity
pip install tqdm

python -m spacy download en_core_web_sm

python linguistic_features.py

module purge