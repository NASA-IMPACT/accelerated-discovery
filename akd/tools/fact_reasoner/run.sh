#!/bin/bash

# use the preprocessed dataset (decontextualized atoms and contexts already retrieved)
# models: granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct
d=/home/radu/git/fm-factual/data/bio-labeled-ChatGPT-langchain-doc.jsonl
o=/home/radu/git/fm-factual/results

# Run FS0
# for m in granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct; do
#     python fm_factual/fact_score.py \
#         --labeled_dataset $d \
#         --dataset_name bio_gold \
#         --model $m \
#         --service_type langchain \
#         --no_contexts \
#         --output_dir $o >& log_fs0_langchain_${m}.txt
# done

# Run FS
# for m in granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct; do
#     python fm_factual/fact_score.py \
#         --labeled_dataset $d \
#         --dataset_name bio_gold \
#         --model $m \
#         --service_type langchain \
#         --output_dir $o >& log_fs_langchain_${m}.txt
# done

# Run FR
# for m in granite-3.0-8b-instruct llama-3.1-70b-instruct llama-3.2-11b-instruct mixtral-8x22b-instruct llama-3.2-90b-instruct; do
#     for v in 1 2 3; do
#         python fm_factual/fact_reasoner.py \
#             --labeled_dataset $d \
#             --use_merlin \
#             --dataset_name bio_gold \
#             --model $m \
#             --service_type langchain \
#             --version $v \
#             --output_dir $o >& log_fr${v}_langchain_${m}.txt
#     done
# done


m=llama-3.2-11b-instruct # there is a bug with factreasoner.score fr2
# m=mixtral-8x7b-instruct
# python fm_factual/fact_score.py \
#     --labeled_dataset $d \
#     --dataset_name bio_gold \
#     --model $m \
#     --service_type langchain \
#     --no_contexts \
#     --output_dir $o >& log_fs0_langchain_${m}.txt

# python fm_factual/fact_score.py \
#     --labeled_dataset $d \
#     --dataset_name bio_gold \
#     --model $m \
#     --service_type langchain \
#     --output_dir $o >& log_fs_langchain_${m}.txt

for v in 3; do
    python fm_factual/fact_reasoner.py \
        --labeled_dataset $d \
        --use_merlin \
        --dataset_name bio_gold \
        --model $m \
        --service_type langchain \
        --version $v \
        --output_dir $o >& log_fr${v}_langchain_${m}.txt
done
