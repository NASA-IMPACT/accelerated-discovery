# coding=utf-8
# Copyright 2023-present the International Business Machines.Gerhard Fischer
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import math
import pandas as pd
import numpy as np

def to_csv(input_file: str, output_file: str):
    """
    Convert a jsonl file into a csv file.
    """
    
    assert ".jsonl" in input_file, f"Input file must be a jsonl file."
    assert ".csv" in output_file, f"Output file must be a csv file."

    with open(input_file, "r") as f:
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']
        df_inter['json_element'].apply(json.loads)
        df_results = pd.json_normalize(df_inter['json_element'].apply(json.loads))

        print(df_results)
        output_file = os.path.join(output_dir, f"{results_file}.csv")
        df_results.to_csv(output_file)

    f.close()

def stats(input_file: str):
    """
    Compute the statistics from the input file (a jsonl file).
    """

    assert ".jsonl" in input_file, f"Input file must be a jsonl file."
    print(f"Stats for: {input_file}")

    with open(input_file, "r") as f:
        lines = f.read().splitlines()
        df_inter = pd.DataFrame(lines)
        df_inter.columns = ['json_element']
        df_inter['json_element'].apply(json.loads)
        df_results = pd.json_normalize(df_inter['json_element'].apply(json.loads))

        # Compute the median number of atoms K
        K = df_results['num_atoms'].median()      # median number of atoms

        # Compute the absolute error, squared error.
        df_results['AE'] = abs(df_results['factuality_score'] - df_results['gold_factuality_score'])
        df_results['SE'] = df_results['AE'].apply(lambda x: math.pow(x, 2))
        df_results['precision'] = df_results['num_true_atoms'] / df_results['num_atoms']
        df_results['temp'] = df_results['num_true_atoms'] / K
        df_results['recall'] = df_results['temp'].apply(lambda x: min(x, 1.0))
        df_results['PR'] = 2 * df_results['precision'] * df_results['recall']
        df_results['F1K'] = np.where(df_results['PR'] > 0, df_results['PR']/(df_results['precision'] + df_results['recall']), 0)
        df_results['P'] = df_results['true_positive'] / (df_results['true_positive'] + df_results['false_positive'])
        df_results['R'] = df_results['true_positive'] / (df_results['true_positive'] + df_results['false_negative'])
        df_results['F1'] = 2 * (df_results['P']*df_results['R']) / (df_results['P'] + df_results['R'])
        if 'num_uniform_atoms' in df_results.columns:
            df_results['num_contradicted_atoms'] = df_results['num_false_atoms'] - df_results['num_uniform_atoms']

        # Compute the mean and stdev of the relevant columns, such as AE, ...
        fsc = df_results['factuality_score'].mean() # precision (factuality score)
        gfs = df_results['gold_factuality_score'].mean() # gold factuality score
        mae = df_results['AE'].mean()               # mean avg error
        mse = df_results['SE'].mean()               # mean squared error
        mna = df_results['num_atoms'].mean()        # mean number of atoms
        mnc = df_results['num_contexts'].mean()     # mean number of contexts
        if 'num_uniform_atoms' in df_results.columns:
            mca = df_results['num_contradicted_atoms'].mean()
        mtp = df_results['true_positive'].mean()    # true positive
        mtn = df_results['true_negative'].mean()    # true negative
        mfp = df_results['false_positive'].mean()   # false positive
        mfn = df_results['false_negative'].mean()   # false negative
        mta = df_results['num_true_atoms'].mean()   # true atoms
        mfa = df_results['num_false_atoms'].mean()  # false atoms
        gta = df_results['gold_true_atoms'].mean()  # gold true atoms
        dna = df_results['num_atoms'].median()      # median number of atoms
        f1k = df_results['F1K'].mean()
        f1s = df_results['F1'].mean()
        if 'avg_entropy' in df_results.columns:
            ent = df_results['avg_entropy'].mean()

        fsc_std = df_results['factuality_score'].std()
        gfs_std = df_results['gold_factuality_score'].std()
        mae_std = df_results['AE'].std()
        mse_std = df_results['SE'].std()
        mna_std = df_results['num_atoms'].std()
        mnc_std = df_results['num_contexts'].std()
        if 'num_uniform_atoms' in df_results.columns:
            mca_std = df_results['num_contradicted_atoms'].std()
        mtp_std = df_results['true_positive'].std()
        mtn_std = df_results['true_negative'].std()
        mfp_std = df_results['false_positive'].std()
        mfn_std = df_results['false_negative'].std()
        mta_std = df_results['num_true_atoms'].std()
        mfa_std = df_results['num_false_atoms'].std()
        gta_std = df_results['gold_true_atoms'].std()
        f1k_std = df_results['F1K'].std()
        f1s_std = df_results['F1'].std()
        if 'avg_entropy' in df_results.columns:
            ent_std = df_results['avg_entropy'].std()

        # Print the results
        print(f" NA: {int(mna):>4} +- {int(mna_std)}")
        print(f" NC: {int(mnc):>4} +- {int(mnc_std)}")
        print(f" TA: {int(mta):>4} +- {int(mta_std)}")
        print(f" FA: {int(mfa):>4} +- {int(mfa_std)}")
        if 'num_uniform_atoms' in df_results.columns:
            print(f" CA: {int(mca):>4} +- {int(mca_std)}")
        print(f"GTA: {int(gta):>4} +- {int(gta_std)}") # gold true atoms
 
        print(f"MAE: {mae:.2f} +- {mae_std:.2f}")
        print(f"MSE: {mse:.2f} +- {mse_std:.2f}")
        print(f" TP: {int(mtp):>4} +- {int(mtp_std)}")
        print(f" TN: {int(mtn):>4} +- {int(mtn_std)}")
        print(f" FP: {int(mfp):>4} +- {int(mfp_std)}")
        print(f" FN: {int(mfn):>4} +- {int(mfn_std)}")

        print(f" DA  {int(dna):>4} -- median atoms")
        print(f"F1K: {f1k:.2f} +- {f1k_std:.2f}")
        print(f" F1: {f1s:.2f} +- {f1s_std:.2f}")
        print(f" Pr: {fsc:.2f} +- {fsc_std:.2f}")
        print(f"GPr: {gfs:.2f} +- {gfs_std:.2f}")
        
        if 'avg_entropy' in df_results.columns:
            print(f"  E: {ent:.2f} +- {ent_std:.2f}")

        if 'num_uniform_atoms' in df_results.columns:
            mua = df_results['num_uniform_atoms'].mean()
            mua_std = df_results['num_uniform_atoms'].std()
            print(f" UA: {int(mua):>4} +- {int(mua_std)}")


    f.close()

    
if __name__ == "__main__":

    # Read the evaluation results (jsonl) and convert to csv
    
    output_dir = "/home/radu/git/fm-factual/results"
    results_file = "eval_results_factverify_langchain_bio3_granite-3.1-8b-instruct"
    input_file = os.path.join(output_dir, f"{results_file}.jsonl")
    output_file = os.path.join(output_dir, f"{results_file}.csv")

    # to_csv(input_file, output_file)
    stats(input_file)

    print("Done.")
    
  