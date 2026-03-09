#!/bin/bash
# script name: train.sh
#SBATCH --job-name="Train detectors"
#SBATCH --comment="This is for fine-tuning the classifiers for LLM-generated text detection"
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-5:30:00


export ENV_MODE="permanent"
export ENV_NAME="prompting-benchmark"

module load python/3.11.11

pip install pandas
pip install torch
pip install numpy
pip install scikit-learn
pip install evaluate
pip install transformers
pip install accelerate
pip install matplotlib
pip install sentencepiece
pip install protobuf

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type zero_shot_baseline --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type zero_shot_baseline --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type chain_of_thought_zero_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type chain_of_thought_one_shot  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type chain_of_thought_one_shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type style_information_style_example  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type style_information_style_example  --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type style_information_style_example --detection_model  xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type style_information_style_example --detection_model  xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type style_information_style_example --detection_model  xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type style_information_style_example --detection_model  xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type style_information_style_example --detection_model  xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type style_information_style_example --detection_model  xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type style_information_style_example --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type style_information_style_example --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type style_information_style_example --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type self-refine --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type self-refine --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model llama3.3 --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
python train_classifiers.py --dataset_type news --generation_model llama3.3 --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model llama3.3 --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
python train_classifiers.py --dataset_type qa --generation_model llama3.3 --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen14b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen14b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen14b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen14b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen32b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen32b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen32b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen32b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model qwen72b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model qwen72b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model qwen72b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model qwen72b --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model solar --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model solar --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model solar --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model solar --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model mistral --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model mistral --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model mistral --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model mistral --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base

#python train_classifiers.py --dataset_type abstracts --generation_model deepseek --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type news --generation_model deepseek --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type reviews --generation_model deepseek --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base
#python train_classifiers.py --dataset_type qa --generation_model deepseek --prompt_type in_context_learning_3-shot --detection_model xlm-roberta-base


module purge