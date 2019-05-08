# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:25:24 2018
@author: jlibor
"""
### Aktuelle Version als Hilfe ausgeben
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch
import deepdish as dd
import torchvision
import functools
import pandas as pd
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer       # python 2
#from http.server import BaseHTTPRequestHandler, HTTPServer        # python 3
import json
#from compute_embedding_snack import compute_graph
import time, threading
import requests
from random import uniform
import python_code.initialization as init
import python_code.communication as communication
from python_code.label_generation import svm_k_nearest_neighbors
import python_code.train as train
from python_code.model import MapNet
from python_code.aux import scale_to_range
import pickle


# initialize global dataset information (image ids, features, embedding) and network
dataset_name = 'Wikiart_Elgammal_EQ_artist_train'
data_dir = './dataset_info'
data_info_file = os.path.join(data_dir, 'info_{}.h5'.format(dataset_name))
label_file = None
impath = None

feature_dim = 512
projection_dim = 2
limits = (-15, 15)
max_display = 1000       # show at most max_display images in interface
print('Initialize...')
initializer = init.Initializer(dataset_name, impath=impath, info_file=data_info_file, feature_dim=feature_dim)
initializer.initialize(dataset=impath is not None, is_test=dataset_name.endswith('_test'))
initial_data = initializer.get_data_dict(normalize_features=True)
if label_file is not None:
    label_data = dd.io.load(label_file)
    if not np.all(label_data['image_id'] == initial_data['image_id']):
        raise RuntimeError('IDs in label_file do not match IDs in data_dict.')
    label_none = label_data['labels_none']
    label_svm = label_data['labels_svm']
    generated_svm = label_svm[(label_none == 'None') & (label_svm != 'None')]
    label_area = label_data['labels_area']
    generated_area = label_area[(label_none == 'None') & (label_area != 'None')]
    # add labels to info_file
    info_df = initial_data['info']
    for col, data in zip(['labeled', 'generated_svm', 'generated_area'], [label_none, generated_svm, generated_area]):
        info_df[col] = None
        info_df.loc[label_data['image_id'], col] = data

print('Done.')
graph_df = None
index_to_id = None
_svm_temp = None
net = None
experiment_id = 'TEST_artist2'#'{}_{}'.format(time.strftime('%m-%d-%H-%M'), dataset_name)
StartTime = time.time()

class GlobalVarSaver():
    """Namespace container class"""
    def __init__(self):
        pass


def reset():
    """Reset the global variables."""
    global dataframe, net, experiment_id
    dataframe = None
    net = None
    experiment_id = 'TEST_artist2'  # '{}_{}'.format(time.strftime('%m-%d-%H-%M'), dataset_name)


def generate_labels_and_weights():
    global _svm_temp, graph_df, initial_data

    labels, weights = graph_df.loc[initial_data['image_id'], ('group', 'weight')].values.transpose()
    # predict the labels using the svms
    for i, (svm, labelname, threshold) in enumerate(_svm_temp['svms']):
        print('{}/{}'.format(i+1, len(_svm_temp['svms'])))
        scores = svm.predict_proba(initial_data['features'])[:, 1]      # positive class only
        group_ids = np.repeat(labelname, len(labels))
        labels, weights = communication.update_labels_and_weights(labels, weights, group_ids, scores,
                                                                  threshold=threshold)

    return labels, weights


def update_coordinates(projection, socket_id, epoch=None):
    global graph_df, initial_data, limits
    # convert indexing of projection to
    min_epoch = 10
    if epoch is None or epoch >= min_epoch:
        ids = [index_to_id[k] for k in sorted(index_to_id.keys())]

        map_index = map(lambda x: initial_data['image_id'].index(x), ids)

        small_graph = graph_df.loc[np.array(initial_data['image_id'])[map_index]]
        x, y = scale_to_range(projection[map_index, 0], limits[0], limits[1]), \
               scale_to_range(projection[map_index, 1], limits[0], limits[1])
        small_graph.loc[:, ('x', 'y')] = zip(x, y)
        payload = communication.graph_df_to_json(small_graph, socket_id=socket_id)
        communication.send_payload(payload)


def store_projection(model, weightfile=None, use_gpu=True, socket_id=None):
    global graph_df, initial_data
    if weightfile is not None:
        state_dict = torch.load(weightfile, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict)
    projection = train.evaluate_model(model.forward, data=initial_data['features'],
                                      batch_size=2000, use_gpu=use_gpu, verbose=False)
    projection = projection.numpy()

    if socket_id is not None:
        update_coordinates(projection, socket_id=socket_id)

    x = scale_to_range(projection[:, 0], limits[0], limits[1])
    y = scale_to_range(projection[:, 1], limits[0], limits[1])

    graph_df.loc[initial_data['image_id'], ('x', 'y')] = zip(x, y)


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
        global graph_df, index_to_id, _svm_temp
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
                                                   info_df=initial_data['info'],
                                                   coordinate_range=limits)

            graph_json = communication.graph_df_to_json(graph_df, max_elements=max_display)
            index_to_id = communication.make_index_to_id_dict(graph_json)

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
            # data = json.dumps({'p': p, 'n': n}).encode()
            data = json.dumps(data).encode()
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
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)

            # convert body to list
            data = json.loads(str(body).decode('utf-8'))  # python 2
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

            group_id = int(data['groupId'])
            thresh = float(data['threshold'])
            if 'negatives' not in data.keys():          # first iteration
                # reset _svm_temp
                _svm_temp = {'positives': data['positives'],
                             'negatives': set([]),              # allow no duplicates
                             'svms': [] if _svm_temp is None else _svm_temp['svms']}
                _svm_temp['svms'].append(None)      # empty entry to save new svm and group_id to

            else:
                _svm_temp['positives'] = data['positives']      # overwrite positives
                _svm_temp['negatives'].update(data['negatives'])
                _svm_temp['negatives'] = _svm_temp['negatives'].difference(_svm_temp['positives'])      # in the unlikely case that a previous negative was somehow labeled as a positive by user

            # only operate on data displayed in interface
            displayed_ids = index_to_id.values()
            displayed_idcs = map(initial_data['image_id'].index, displayed_ids)             # convert displayed indices to indices of all samples
            vectors = initial_data['features'][displayed_idcs]

            # map positive_idcs and negative_idcs to displayed_idcs indexing
            positive_ids = map(lambda x: index_to_id[x], _svm_temp['positives'])
            positive_idcs = map(displayed_ids.index, positive_ids)
            negative_ids = map(lambda x: index_to_id[x], _svm_temp['negatives'])
            negative_idcs = map(displayed_ids.index, negative_ids)

            neighbor_idcs, scores, svm = svm_k_nearest_neighbors(vectors, positive_idcs, negative_idcs,
                                                                 max_rand_negatives=10,
                                                                 k=-1, verbose=False)

            neighbor_ids = map(lambda x: displayed_ids[x], neighbor_idcs)                 # revert displayed_idcs indexing
            id_to_index = dict(zip(index_to_id.values(), index_to_id.keys()))
            neighbor_idcs = map(lambda x: id_to_index[x], neighbor_ids)

            # save user labels
            user_labeled_ids = positive_ids + negative_ids
            labels, weights = graph_df.loc[user_labeled_ids, ('group', 'weight')].values.transpose()
            new_labels = [group_id] * len(positive_ids) + [-group_id] * len(negative_ids)
            new_weights = np.ones(len(user_labeled_ids))
            labels, weights = communication.update_labels_and_weights(labels, weights, new_labels, new_weights)
            graph_df.loc[user_labeled_ids, ('group', 'weight')] = zip(labels, weights)

            # save svm info for label prediction later
            _svm_temp['svms'][-1] = (svm, group_id, thresh)

            # make json
            return_dict = {'group': _svm_temp['positives'],
                           'neighbours': dict(zip(neighbor_idcs, 1. - scores))}     # reverse scores
            return_dict = json.dumps(return_dict).encode()
            self.wfile.write(return_dict)  # body zurueckschicken


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

            plot_fn = functools.partial(update_coordinates, socket_id=self.socket_id)

            self.wfile.write('update_embedding started for ' + str(self.socket_id))  # body zurueckschicken


            # TODO: USE REAL USER LABELS INSTEAD OF GROUND TRUTH
            # # infer the current user labels
            # display_graph_df = communication.json_graph_to_df(body['nodes'])
            # display_graph_df.index = map(lambda x: index_to_id[x], display_graph_df.index.values.astype(int))
            #
            # # update the labels
            # labels, weights = graph_df.loc[display_graph_df.index, ('group', 'weight')].values.transpose()
            # new_labels = display_graph_df['group'].values
            # new_weights = np.where(new_labels != 0, 1, None)
            # labels, weights = communication.update_labels_and_weights(labels, weights, new_labels, new_weights)
            #
            # graph_df.loc[display_graph_df.index, ('group', 'weight')] = zip(labels, weights)
            #
            # # infer the labels
            # print('Generate Labels...')
            # labels, weights = generate_labels_and_weights()     # these are sorted according to initial_data['image_id']
            # print('Done.')
            #
            # # update the graph
            # graph_df.loc[initial_data['image_id'], ('group', 'weight')] = zip(labels, weights)

            # load ground truth labels
            label_file = './MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_test.hdf5'
            gt_label_df = dd.io.load(label_file)['df']
            gt_label_df.index = gt_label_df['image_id']
            gt_label_df = gt_label_df['artist_name'].dropna()
            label_to_int = {l: i + 1 for i, l in enumerate(set(gt_label_df.values))}
            labels = np.array(map(lambda x: label_to_int[x], gt_label_df.values))
            weights = np.where(labels != 0, 1, None)

            # map labels - just in case
            sorted_index = map(lambda x: initial_data['image_id'].index(x), gt_label_df.index)
            labels, weights = labels[sorted_index], weights[sorted_index]

            graph_df.loc[gt_label_df.index, ('group', 'weight')] = zip(labels, weights)

            # set training and output variables and initialize model
            use_gpu = torch.cuda.is_available()
            weightfile_mapnet = './models/{}_mapnet.pth.tar'.format(experiment_id)
            log_dir = './.log/{}'.format(experiment_id)

            model = MapNet(feature_dim=feature_dim, output_dim=projection_dim)
            if use_gpu:
                model = model.cuda()

            train.train_mapnet(model, initial_data['features'], labels.astype(np.long), weights=weights.astype(np.float32),
                               outpath=weightfile_mapnet, log_dir=log_dir, verbose=False,
                               max_epochs=30, use_gpu=use_gpu, plot_fn=plot_fn)

            store_projection(model, weightfile_mapnet, use_gpu=use_gpu, socket_id=self.socket_id)
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
