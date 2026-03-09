#!/bin/bash
# script name: stylometric_features.sh
#SBATCH --job-name="Linguistic feature extraction"
#SBATCH --comment="Extract stylometric features from the dataset"
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --time=0-35:00:00

export ENV_MODE="permanent"
export ENV_NAME="stylometric-features"

module load python/3.11.11

pip install pandas
pip install spacy
pip install numpy
pip install stylo_metrix
pip install tqdm

python -m spacy download en_core_web_trf

python stylometric_features.py

module purge