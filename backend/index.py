from flask import Flask
from flask import jsonify
from flask import request
from crossdomain import *
import graph

app = Flask(__name__)


@app.route('/search/<path:alg>', methods=['POST', 'GET', 'OPTIONS'])
@crossdomain(origin='*')
def graph_search(alg):
    if request.method == 'POST':
        iters, all_nodes_color, node, all_node_f = graph.receive(
            request.get_json(), alg)
        path = [x.state for x in node.path()]
        # sol = node.solution()
        cost = node.path_cost
        res = jsonify(
            {"iters": iters, "all_nodes_color": all_nodes_color, "path": path, "cost": cost, "all_node_f": all_node_f})
        print(res)
        return res
    else:
        return alg
