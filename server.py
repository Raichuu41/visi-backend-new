# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:25:24 2018
@author: jlibor
"""
### Aktuelle Version als Hilfe ausgeben
import os
import sys
import numpy as np
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer       # python 2
#from http.server import BaseHTTPRequestHandler, HTTPServer        # python 3
import json
#from compute_embedding_snack import compute_graph
import time, threading
import requests
from random import uniform
sys.path.append('MapNetCode')
from initialization import initialize
from communication import make_nodes, read_nodes
from train import train, get_modified, get_neighborhood, dummy_func

import pickle

# initialize global dataset information (image ids, features, embedding) and network
dataset_info = None
net = None

StartTime = time.time()


def update_embedding_handler(socket_id):
    print('action ! -> time : {:.1f}s'.format(time.time()-StartTime))
    nodes = []
    for x in range(0, 2400):
        nodes.append({'id': x, 'x': round(uniform(0, 25), 2), 'y': round(uniform(0, 25))})

    headers = {'content-type': 'application/json'}
    payload = {'nodes': nodes, 'socket_id': socket_id}
    #print(payload)
    response = requests.post("http://localhost:3000/api/v1/updateEmbedding", data=json.dumps(payload), headers=headers)
    print(response)


class SetInterval:
    """
    inspired from https://stackoverflow.com/questions/2697039/python-equivalent-of-setinterval/48709380#48709380
    """
    def __init__(self, interval, action):
        self.socket_id = ''
        self.interval = interval
        self.action = action
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.__set_interval)
        #self.thread.start()
        #self.next_time = 0

    def __set_interval(self):
        next_time = time.time() + self.interval
        while not self.stopEvent.wait(next_time-time.time()):
            next_time += self.interval
            self.action(self.socket_id)

    def start(self):
        print('start timer')
        self.thread.start()

    def cancel(self):
        print('stop timer')
        self.stopEvent.set()


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

id =''

class MyHTTPHandler(BaseHTTPRequestHandler):


    """
    ### MyHTTPHandler beschreibt den Umgang mit HTTP Requests
    """
    #http://donghao.org/2015/06/18/override-the-__init__-of-basehttprequesthandler-in-python/
    def __init__(self, request, client_address, server):
        self.socket_id = ''
        self.inter = SetInterval(0.6, update_embedding_handler)

        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

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
        global dataset_info, net
        if self.path == "/nodes":
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
            print(data)

            # Katjas code goes here
            net, dataset_info = initialize()
            nodes = make_nodes(position=dataset_info['position'],
                               name=dataset_info['name'],
                               label=dataset_info['label'])
            categories = dataset_info['categories']
            data = {'nodes': nodes, 'categories': categories}

            # make json
            data = json.dumps(data).encode()
            self.wfile.write(data)  #body zurueckschicken

        if self.path == "/trainSvm":
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
            print(data)

            # Katjas code goes here
            # p, n = katja_function(data.p, data.n)

            # make json
            # data = json.dumps({p: p, n: n}).encode()
            self.wfile.write(data)  #body zurueckschicken

        if self.path == "/stopSvm":
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
            # p, n = katja_function(data.p, data.n)

            # make json
            #data = json.dumps({p: p, n: n}).encode()
            self.wfile.write("stopped Svm")  #body zurueckschicken

        if self.path == "/updateLabels":
            print("post /updateLabels")
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
            # katja_function(data.p, data.n)

            # make json
            #data = json.dumps({}).encode()
            self.wfile.write(data)  #body zurueckschicken

        if self.path == "/startUpdateEmbedding":
            print("post /startUpdateEmbedding")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            body = json.loads(str(body).decode('utf-8'))  # python 2
            # data = json.loads(str(body, encoding='utf-8'))      # python 3
            #print(body)

            #print(self.socket_id)
            self.socket_id = body['socketId']
            id = body['socketId']
            print(id)
            print(self.socket_id)

            data = read_nodes(body['nodes'])
            ### DEBUG
            # with open('debug_body.pkl', 'w') as f:
            #     pickle.dump({'body': data, 'dataset_info': dataset_info, 'net': net}, f)
            # print('saved body.')
            # return 0
            # with open('debug_body.pkl', 'r') as f:
            #     debug_data = pickle.load(f)
            # data = debug_data['body']
            # dataset_info = debug_data['dataset_info']
            # net = debug_data['net']

            # Katjas code goes here
            new_position = np.stack([data['x'], data['y']], axis=1)
            old_position = dataset_info['position']

            # idx_modified = get_modified(old_position, new_position)
            # idx_old_neighbors = get_neighborhood(old_position, idx_modified)
            # idx_new_neighbors = get_neighborhood(new_position, idx_modified)

            idx_modified = get_modified(old_position, new_position)
            idx_old_neighbors = np.arange(50, 60)
            idx_new_neighbors = np.arange(50, 300)

            print('Modified nodes: {}'.format([dataset_info['name'][idx_modified]]))

            dummy_func(net, dataset_info['feature'], dataset_info['position'],
                  idx_modified, idx_old_neighbors, idx_new_neighbors,
                  lr=1e-4, experiment_id=None, socket_id=self.socket_id,
                  node_id=dataset_info['name'])  # TODO: correct socket ID?


            # train(net, dataset_info['feature'], dataset_info['position'],
            #       idx_modified, idx_old_neighbors, idx_new_neighbors,
            #       lr=1e-4, experiment_id=None, socket_id=self.socket_id, node_id=dataset_info['name'])        # TODO: correct socket ID?

            # TODO was ist wenn das mehrfach gestartet wird
            # self.inter = SetInterval(0.6, update_embedding_handler, id)
            # self.inter.socket_id = id
            # self.inter.start()
            # t = threading.Timer(5, self.inter.cancel)
            # t.start()

            # make json
            # data = json.dumps({}).encode()
            self.wfile.write('update_embedding started for ' + str(self.socket_id))  # body zurueckschicken

        if self.path == "/stopUpdateEmbedding":
            print("post /stopUpdateEmbedding")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            # content_len = int(self.headers['Content-Length'])
            # body = self.rfile.read(content_len)

            # convert body to list
            # data = json.loads(str(body).decode('utf-8'))  # python 2
            # data = json.loads(str(body, encoding='utf-8'))      # python 3
            # print(data)

            # Katjas code goes here
            print(self.socket_id)
            self.inter.cancel()
            #t = threading.Timer(5, self.inter.cancel)
            #t.start()

            #print(id)
            # make json
            # data = json.dumps({}).encode()
            self.wfile.write('update_embedding stopped for ' + str(self.socket_id))  # body zurueckschicken

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
