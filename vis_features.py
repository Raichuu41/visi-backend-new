# -*- coding: utf-8 -*-
"""
@author: kschwarz
"""
import os
import sys
import deepdish as dd
import numpy as np
import pandas as pd
import argparse
from snack import snack_embed
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer       # python 2
#from http.server import BaseHTTPRequestHandler, HTTPServer        # python 3
import json


parser = argparse.ArgumentParser(description='Visualize feature embedding and labels from dataset in browser.')
# wikiart artist dataset
parser.add_argument('--info_file', type=str, help='hdf5 file containing dataframe of dataset (image names and labels)')
parser.add_argument('--feature_file', type=str, help='hdf5 file containing features of dataset (image names and features)')

args = parser.parse_args()
args.embedding_func = snack_embed  # function to use for low dimensional projection / embedding
args.kwargs_emb_func = {'contrib_cost_tsne': 100, 'contrib_cost_triplets': 0.,
                        'perplexity': 30, 'theta': 0.5, 'no_dims': 2}  # kwargs for embedding_func


def compute_graph(feature_file, info_file, embedding_func, **kwargs):
    def initial_embedding(features, embedding_func, **kwargs):
        print('Compute embedding...')
        tic = time.time()
        embedding = embedding_func(np.stack(features).astype(np.double),
                                   triplets=np.zeros((1, 3), dtype=np.long),  # dummy values
                                   **kwargs)
        toc = time.time()
        print('Done. ({:2.0f}min {:2.1f}s)'.format((toc - tic) / 60, (toc - tic) % 60))
        return embedding
    def construct_nodes(names, positions, labels=None):
        def df_to_dict(dataframe):
            dct = dataframe.to_dict(orient='index')
            # replace any values in arrays by lists
            array_keys = [k for k, v in dct[0].items() if isinstance(v, np.ndarray)]
            for d in dct:
                for k in array_keys:
                    dct[d][k] = list(dct[d][k])
            return dct

        if labels is None:
            labels = [None] * len(names)
        nodes = pd.DataFrame({'name': names, 'x': positions[0], 'y': positions[1], 'labels': list(labels)})
        nodes = df_to_dict(nodes)
        for n in nodes:
            nodes[n]['labels'] = list(nodes[n]['labels'])
        return nodes

    # read info file and feature file
    info_dict = dd.io.load(info_file)['df']
    image_names = info_dict['image_id'].values.astype(str)
    features = dd.io.load(feature_file)['features']
    assert (image_names == dd.io.load(feature_file)['image_names']).all(), \
        'Features do not match image names in info file.'
    # normalize features
    features = (features - np.mean(features, axis=1, keepdims=True)) / np.linalg.norm(features, axis=1, keepdims=True)
    categories = ['artist_name', 'style', 'genre', 'technique', 'century']
    labels = np.stack(info_dict[k].values for k in categories if k in info_dict.keys()).transpose()

    embedding = initial_embedding(features, embedding_func, **kwargs)     # initialise prev_embedding with standard tsne
    nodes = construct_nodes(image_names, embedding.transpose(), labels)

    return nodes, categories


## MyHTTPHandler beschreibt den Umgang mit HTTP Requests
class MyHTTPHandler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', self.headers['origin'])
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-type')

        self.end_headers()


    def do_POST(self):
        """
        definiert den Umgang mit POST Requests
        Liest den Body aus - gibt in zum konvertieren weiter

        """
        # global embedding, nodes, usr_labeled_idcs
        if(self.path == "/nodes"):
            print("post /nodes")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            #self.send_header('Access-Control-Allow-Origin', self.headers['origin'])
            self.end_headers()

            # Katjas code goes here
            nodes, categories = compute_graph(args.feature_file, args.info_file,
                                              args.embedding_func, **args.kwargs_emb_func)
            # make json
            data = json.dumps({'nodes': nodes, 'categorys': categories}).encode()
            self.wfile.write(data)  #body zurueckschicken

if __name__ == "__main__":
    # config
    HOST_NAME = ""
    PORT_NUMBER = 8000
    try:
        http_server = HTTPServer((HOST_NAME, PORT_NUMBER), MyHTTPHandler)
        print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
        http_server.serve_forever()
    except KeyboardInterrupt:
        print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
http_server.socket.close()