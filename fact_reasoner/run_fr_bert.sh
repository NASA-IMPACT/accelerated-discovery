#!/bin/bash

# 1st argument is the version (1, 2, 3)
# 2nd argument is the dataset name
# 3rd argument is the service type
# 4th argument is the input file

# Run FactReasoner

# use the preprocessed dataset (decontextualized atoms and contexts already retrieved)
# models: granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct
# d=/home/radu/git/fm-factual/data/askhist-unlabeled-langchain-doc.jsonl

o=/home/radu/git/fm-factual/results
p=/home/radu/git/fm-factual/lib/merlin

v=$1
d=$2
s=$3
f=$4

python fm_factual/fact_reasoner.py \
    --input_file $f \
    --dataset_name $d \
    --model granite-3.1-8b-instruct \
    --service_type $s \
    --version $v \
    --nli_prompt_version v1 \
    --merlin_path $p \
    --bert_nli \
    --use_priors \
    --output_dir $o >& log_fr${v}_${s}_${d}_roberta.txt
