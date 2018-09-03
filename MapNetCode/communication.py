import numpy as np
import pandas as pd
import json
import requests


def make_nodes(position, name=None, label=None):
    """Args:
        names (list or np.ndarray): names of nodes, e.g. image ID
        position (2 x N np.ndarray): x,y coordinates of datapoints
        label (list or np.ndarray), optional: labels for datapoints (to assign various labels to one point choose list
            of lists or np.ndarray of shape N x L, where L is the number of labels)"""
    if label is None:
        label = [None] * position.shape[1]
    elif isinstance(label, np.ndarray):  # label is array with labels --> convert to list
        label = label.tolist()
    elif isinstance(label[0], np.ndarray):  # label is list of array labels --> convert to list
        label = map(np.ndarray.tolist, label)
    nodes = pd.DataFrame({'x': position[0], 'y': position[1], 'labels': list(label)})
    if name is not None:
        nodes['name'] = name
    return nodes.to_dict(orient='index')


def read_nodes(json_nodes):
    return pd.DataFrame.from_dict(json_nodes, orient='index')


def send_payload(nodes, socket_id):
    headers = {'content-type': 'application/json'}
    payload = {'nodes': nodes, 'socket_id': socket_id}
    #print(payload)
    requests.post("http://localhost:3000/api/v1/updateEmbedding", data=json.dumps(payload), headers=headers)