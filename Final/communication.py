"""This script includes all functions to communicate with the web interface
and infer information from its return values."""
import pandas as pd
import numpy as np


def read_json_nodes(json_nodes):
    df = pd.DataFrame.from_dict(json_nodes, orient='index')
    df.index = df['index']
    return df.sort_index()


def detect_movement(old_position, new_position, tol=0.):
    return np.where(np.linalg.norm(old_position - new_position, axis=1) > tol)[0]


def send_nodes_to_interface(nodes, socket_id, categories=None):
    headers = {'content-type': 'application/json'}
    payload = {'nodes': nodes, 'socket_id': socket_id}
    if categories is not None:
        payload['categories'] = categories
    #print(payload)
    requests.post("http://localhost:3000/api/v1/updateEmbedding", data=json.dumps(payload), headers=headers)


