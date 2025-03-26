#!/bin/bash

# 1st argument is the model name
# 2nd argument is the dataset name
# 3rd argument is the service type (google, langchain)
# 4th argument is the input file

# Run FactScore

# ./run_fs.sh granite-3.1-8b-instruct askhist /home/radu/git/fm-factual/data/askhist-unlabeled-langchain-doc.jsonl

# use the preprocessed dataset (decontextualized atoms and contexts already retrieved)
# models: granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct
# d=/home/radu/git/fm-factual/data/askhist-unlabeled-langchain-doc.jsonl

o=/home/radu/git/fm-factual/results

m=$1
d=$2
s=$3
f=$4

python fm_factual/fact_score.py \
    --input_file $f \
    --dataset_name $d \
    --model $m \
    --service_type $s \
    --prompt_version v1 \
    --binary_output \
    --output_dir $o >& log_fs1_${s}_${d}_${m}.txt
