# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:25:24 2018
@author: jlibor
"""
### Aktuelle Version als Hilfe ausgeben
import os
import sys
import time
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer       # python 2
#from http.server import BaseHTTPRequestHandler, HTTPServer        # python 3
import json
from graph_embedding import compute_graph, extract_labels
import numpy as np
from svm import svm_iteration, local_update
# from compute_embedding import compute_graph, learn_svm, local_embedding, train_global_svm, \
#     local_embedding_with_all_positives, write_final_svm_output

# Katja's global variables
embedding = None
nodes = None
usr_labeled_idcs = None


def format_return_graph(graph):
    def format_return_nodes(nodes):
        formatted_nodes = {}
        for k in nodes.keys():
            n = nodes[k].copy()
            idx = n.pop('index')
            formatted_nodes[idx] = n
        return formatted_nodes
    if graph['nodes'] is not None and len(graph['nodes']) != 0:
        graph['nodes'] = format_return_nodes(graph['nodes'])


"""
def format_string(graph):
    s = str(graph)
    s = s.replace("'", '"').replace(': ', ':').replace('False', 'false').replace('True', 'true')\
        .replace(', ', ',').replace(':u"', ':"')
    return s
"""

"""
### dev Server
def get_graph(userData = []):
    filename = "data/response_data.txt"
    with open(filename, "rb") as f:
        return f.read()
"""

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
        global embedding, nodes, usr_labeled_idcs
        if(self.path == "/nodes"):
            print("post /nodes")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            #self.send_header('Access-Control-Allow-Origin', self.headers['origin'])
            self.end_headers()

            # get body from request
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            data = json.loads(str(body).decode('utf-8'))  # python 2
            #data = json.loads(str(body, encoding='utf-8'))      # python 3
            # print(data)
            format_return_graph(data)

            # Katjas code goes here
            nodes, categories = compute_graph(data, embedding)
            embedding = None    # enable update from graph values --> new positions set by user
            # data = multiclass_embed(data)

            # make json
            data = json.dumps({'nodes': nodes, 'categorys': categories}).encode()
            self.wfile.write(data)  #body zurueckschicken

        if(self.path == "/trainSvm"):
            print("post /trainsvm")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            data = json.loads(str(body).decode('utf-8'))  # python 2
            #data = json.loads(str(body, encoding='utf-8'))      # python 3
            # print(data)

            # Katjas code goes here
            usr_labeled_idcs = np.concatenate([data['p'], data['n']])
            p, n = svm_iteration(data['p'], data['n'], data['count'], {'nodes': nodes})     # TODO: REMOVE NASTY HACK WITH NODES
            # p, n, t = learn_svm(data['p'], data['n'], data['count'])

            # make json
            data = json.dumps({'p': p, 'n': n, 't': []}).encode()
            self.wfile.write(data)  #body zurueckschicken

        if(self.path == "/stopSvm"):
            print("post /stopSvm")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            #content_len = int(self.headers['Content-Length'])
            #body = self.rfile.read(content_len)

            # convert body to list
            #data = json.loads(str(body).decode('utf-8'))  # python 2
            #data = json.loads(str(body, encoding='utf-8'))      # python 3
            #print(data)

            # Katjas code goes here
            embedding, local_positives = local_update({'nodes': nodes}, usr_labeled_idcs)              # TODO: REMOVE NASTY HACK WITH NODES and train idcs

            # make json
            data = json.dumps({'group': local_positives}).encode()
            self.wfile.write(data)  #body zurueckschicken

        if(self.path == "/updateLabels"):
            print("post /updateLabels")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            data = json.loads(str(body).decode('utf-8'))  # python 2
            #data = json.loads(str(body, encoding='utf-8'))      # python 3
            #print(data)

            format_return_graph(data)
            # Katjas code goes here
            nodes, categories = extract_labels(data)

            # make json
            data = json.dumps({'nodes': nodes, 'categorys': categories}).encode()
            self.wfile.write(data)  #body zurueckschicken


        return

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