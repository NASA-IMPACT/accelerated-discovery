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
import pandas as pd

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.readwrite import UAIWriter, UAIReader

if __name__ == "__main__":

    # Debugging inference in Markov networks
    
    output_dir = "/home/radu/git/fm-factual/examples"
    filename = os.path.join(output_dir, "markov_network.uai")

    reader = UAIReader(filename)
    # model = reader.get_model()
    variables = reader.get_variables()
    edges = reader.get_edges()
    tables = reader.get_tables()
    print(f"{len(variables)}: {variables}")
    print(f"{len(edges)}: {edges}")
    print(f"{len(tables)}: {tables}")

    G = MarkovNetwork()
    G.add_nodes_from(variables)
    G.add_edges_from(edges)

    for t in tables:
        scope = t[0]
        card = [2] * len(scope)
        values = [float(v) for v in t[1]]
        f = DiscreteFactor(scope, card, values)
        G.add_factors(f)

    print(G)

    print(f"Exact posterior marginal for: {variables}")
    
    results = {}
    inference = VariableElimination(G)
    for qvar in variables:
        print(f"processing {qvar}")
        phi_query = inference.query(
            variables=[qvar],
            elimination_order="MinFill",
            joint=False,
            show_progress=False
        )

        # inference = BeliefPropagationWithMessageParsing(model.to_factor_graph())
        # print(f"Belief Propagation for: {variables}")
        # phi_query = inference.query(variables, evidence={})
        results.update(phi_query)

    print(results)
    print("Done.")
    