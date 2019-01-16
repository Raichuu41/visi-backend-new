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
import python_code.initialization as init
import python_code.communication as communication
from python_code.aux import scale_to_range
import pickle


# initialize global dataset information (image ids, features, embedding) and network
dataset_name = 'Wikiart_Elgammal_test'
feature_dim = 512
limits = (-15, 15)
max_display = 100       # show at most max_display images in interface
data_info_file = './MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_test.hdf5'
print('Initialize...')
initial_data, initial_data_info = init.get_data(dataset_name=dataset_name, impath=None,
                                                normalization_file=None, info_file=data_info_file,
                                                feature_dim=feature_dim)
print('Done.')
dataframe = None
net = None
experiment_id = 'TEST'#'{}_{}'.format(time.strftime('%m-%d-%H-%M'), dataset_name)

StartTime = time.time()


def reset():
    """Reset the global variables."""
    global dataframe, net, experiment_id
    dataframe = None
    net = None
    experiment_id = 'TEST'  # '{}_{}'.format(time.strftime('%m-%d-%H-%M'), dataset_name)


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
        global dataframe
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
            # print(data)

            # Katjas code goes here
            reset()
            graph_df = communication.make_graph_df(image_ids=initial_data['image_id'],
                                                   projection=initial_data['projection'],
                                                   info_df=initial_data_info,
                                                   coordinate_range=limits)

            graph_json = communication.graph_df_to_json(graph_df, max_elements=max_display)
            self.wfile.write(graph_json)  #body zurueckschicken
            # print(graph_json)


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

        if self.path == "/getGroupNeighbours":
            print("post /getGroupNeighbours")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            # get body from request
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            data = json.loads(str(body).decode('utf-8'))  # python 2
            # data = json.loads(str(body, encoding='utf-8'))      # python 3

            print(data)



            # # make json
            # data['neighbours'] = {neigh: score for neigh, score in zip(_neighbors['positive'], scores)}
            # print(len(data['positives']))
            data['group'] = list(data['positives'])
            data = json.dumps(data).encode()
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

            data = read_nodes(body['nodes'])
            self.wfile.write('update_embedding started for ' + str(self.socket_id))  # body zurueckschicken

            # Katjas code goes here
            new_position = np.stack([data['x'], data['y']], axis=1)
            new_position = dataset_info['inverse_scale_func'](new_position)
            old_position = dataset_info['position']

            idx_modified = get_modified(old_position[:max_display], new_position, tol=dataset_info['inverse_scale_func'](np.array([3, ])))
            if len(idx_modified) == 0 or len(idx_modified) == len(dataset_info['name']):       # TODO: fix recall bug
                print('Modified {} samples. - Invalid for training.')
                return 0
            # idx_old_neighbors = get_neighborhood(old_position[:N], idx_modified)
            # idx_new_neighbors = get_neighborhood(new_position[:N], idx_modified)
            idx_old_neighbors, _ = mutual_k_nearest_neighbors(old_position[:max_display], idx_modified, k=50)
            idx_new_neighbors = get_neighborhood(new_position[:max_display], idx_modified)

            print('Train MapNet using {} positives and {} negatives.'.format(len(idx_modified) + len(idx_new_neighbors),
                                                                             len(_neighbors['negative'])))

            # net, dataset_info = initialize(dataset=dataset, experiment_id=experiment_id, batch_size=100,
            #                                lr=1e-4)
            # experiment_id = dataset_info['experiment_id']
            # # introduce scale factor
            # limits = (-15, 15)
            # pmax, pmin = dataset_info['position'].max(), dataset_info['position'].min()
            # dataset_info['scale_func'] = lambda x: np.divide((limits[1] - limits[0]) * (x.copy() - pmin), pmax - pmin) + \
            #                                        limits[0]
            # dataset_info['inverse_scale_func'] = lambda x: np.divide((x.copy() - limits[0]) * (pmax - pmin),
            #                                                          limits[1] - limits[0]) + pmin
            # if N is None:
            #     N = len(dataset_info['name'])
            new_position = train(net, dataset_info['feature'][:max_display], dataset_info['name'].values.astype(str)[:max_display],
                                 old_position[:max_display], new_position,
                                 idx_modified, idx_old_neighbors, idx_new_neighbors,
                                 _neighbors['negative'],
                                 categories=dataset_info['categories'], label=dataset_info['label'][:max_display],
                                 lr=1e-3, experiment_id=experiment_id, socket_id=self.socket_id,
                                 scale_func=dataset_info['scale_func'])

            dataset_info['position'] = new_position
            _neighbors = {'positive': [], 'negative': []}
            return 0

            # TODO was ist wenn das mehrfach gestartet wird
            # self.inter = SetInterval(0.6, update_embedding_handler, id)
            # self.inter.socket_id = id
            # self.inter.start()
            # t = threading.Timer(5, self.inter.cancel)
            # t.start()

            # make json
            # data = json.dumps({}).encode()

        if self.path == "/stopUpdateEmbedding":
            print("post /stopUpdateEmbedding")
            ### POST Request Header ###
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()

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
