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
from http.server import BaseHTTPRequestHandler, HTTPServer  # python 3
import json
# from compute_embedding_snack import compute_graph
import time, threading
import requests
from random import uniform
import python_code.initialization as init
import python_code.communication as communication
from python_code.label_generation import svm_k_nearest_neighbors
import python_code.train as train
from python_code.model import MapNet, mapnet
from python_code.aux import scale_to_range, load_weights
import pickle

# from IPython import embed

N_LAYERS = 0
DATA_DIR = sys.argv[1]
IMPATH = None  # should not be needed, since all features should be precomputed
FEATURE_DIM = 512
PROJECTION_DIM = 2
LIMITS = (-15, 15)
MAX_DISPLAY = None  # show at most MAX_DISPLAY images in interface (None == don't cut anything)
START_TIME = time.time()
DEVICE = 2

SPLASH = """
+-------------------------------------------+
| #   # ##### ##### ##### ##### #   # ##### |
| #   #   #   #       #   #      # #  #   # |
|  # #    #   #####   #   ###     #   ##### |
|  # #    #       #   #   #      # #  #     |
|   #   ##### ##### ##### ##### #   # #     |
+-------------------------------------------+
"""
# global variables
initial_datas = {}
user_datas = {}


# initialize global dataset information (image ids, features, embedding) and network
# label_file = None

class UserData:
    """
    stores data per user, which was previously handled globally
    """

    def __init__(self, user_id):
        print(f"[[ creating user {user_id} ]]")  # DEBUG!
        self.user_id = user_id
        self.dataset = None  # dataset the user is working on
        self.svm_ids = []
        self.svm_negs = set()

        # LEGACY: remove asap
        self.index_to_id = None
        self._svm_temp = None
        self.graph_df = None

    def __repr__(self):
        return f"<UserData({self.user_id}) at {id(self)}>"


def initialize_dataset(dataset_name):
    print(f'Initialize {dataset_name}...')
    data_info_file = os.path.join(DATA_DIR, f'{dataset_name}.json')
    initializer = init.Initializer(dataset_name, impath=IMPATH, info_file=data_info_file,
                                   feature_dim=FEATURE_DIM, outdir=DATA_DIR)
    initializer.initialize(dataset=IMPATH is not None, is_test=dataset_name.endswith('_test'), raw_features=True)
    initial_datas[dataset_name] = initializer.get_data_dict(normalize_features=True)
    # embed() #DEBUG!!
    print(f'Done [{dataset_name}].')


def dataset_id_to_name(ds_id):
    "not needed anymore"
    d = {'002': 'Wikiart_artist49_images',
         '003': 'AwA2_vectors_train',
         '004': 'AwA2_vectors_test',
         '005': 'STL_label_train',
         '006': 'STL_label_test',
         '011': 'STL_label_test_random',
         '001': 'Wikiart_Elgammal_EQ_artist_test',
         '008': 'Wikiart_Elgammal_EQ_artist_train',
         '009': 'Wikiart_Elgammal_EQ_genre_train',
         '010': 'Wikiart_Elgammal_EQ_genre_test'
         }

    return d[ds_id]


def generate_projections(model, features):
    if type(model) is str:
        model = load_model(model)
    model.eval()
    model.cuda()
    ds = torch.utils.data.TensorDataset(torch.tensor(features))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    proj = []
    for item in dl:
        item = item[0].cuda()
        fts = model.mapping.forward(item)
        fts = fts / fts.norm(dim=1, keepdim=True)
        proj.append(model.embedder.forward(fts).cpu())
    return torch.cat(proj).detach().numpy()


def load_model(weightfile):
    model = mapnet(N_LAYERS, pretrained=False)
    best_weights = load_weights(weightfile, model.state_dict())
    model.load_state_dict(best_weights)
    return model


def generate_labels_and_weights():
    """
    used in "/startUpdateEmbedding"
    """
    global _svm_temp, graph_df, initial_data

    labels, weights = graph_df.loc[initial_data['image_id'], ('group', 'weight')].values.transpose()
    # predict the labels using the svms
    for i, (svm, labelname, threshold) in enumerate(_svm_temp['svms']):
        print(f'{i + 1}/{len(_svm_temp["svms"])}')
        scores = svm.predict_proba(initial_data['features'])[:, 1]  # positive class only
        group_ids = np.repeat(labelname, len(labels))
        labels, weights = communication.update_labels_and_weights(labels, weights, group_ids, scores,
                                                                  threshold=threshold)

    return labels, weights


def update_coordinates(projection, socket_id, epoch=None):
    """
    used in `/startUpdateProjection` and store_projection()
    """
    global graph_df, initial_data, LIMITS
    # convert indexing of projection to
    min_epoch = 10
    if epoch is None or epoch >= min_epoch:
        ids = [index_to_id[k] for k in sorted(index_to_id.keys())]

        map_index = map(lambda x: initial_data['image_id'].index(x), ids)

        small_graph = graph_df.loc[np.array(initial_data['image_id'])[map_index]]
        x, y = scale_to_range(projection[map_index, 0], LIMITS[0], LIMITS[1]), \
               scale_to_range(projection[map_index, 1], LIMITS[0], LIMITS[1])
        small_graph.loc[:, ('x', 'y')] = zip(x, y)
        payload = communication.graph_df_to_json(small_graph, socket_id=socket_id)
        communication.send_payload(payload)


def store_projection(model, weightfile=None, use_gpu=True, socket_id=None):
    """
    used in `/startUpdateProjection`
    """
    global graph_df, initial_data
    if weightfile is not None:
        state_dict = torch.load(weightfile, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict)
    projection = train.evaluate_model(model.forward, data=initial_data['features'],
                                      batch_size=2000, use_gpu=use_gpu, verbose=False)
    projection = projection.numpy()

    if socket_id is not None:
        update_coordinates(projection, socket_id=socket_id)

    x = scale_to_range(projection[:, 0], LIMITS[0], LIMITS[1])
    y = scale_to_range(projection[:, 1], LIMITS[0], LIMITS[1])

    graph_df.loc[initial_data['image_id'], ('x', 'y')] = zip(x, y)


def update_embedding_handler(socket_id):
    print('action ! -> time : {:.1f}s'.format(time.time() - START_TIME))
    nodes = []
    for x in range(0, 2400):
        nodes.append({'id': x, 'x': round(uniform(0, 25), 2), 'y': round(uniform(0, 25))})

    headers = {'content-type': 'application/json'}
    payload = {'nodes': nodes, 'socket_id': socket_id}
    response = requests.post("http://localhost:3000/api/v1/updateEmbedding", data=json.dumps(payload), headers=headers)
    print(response)


def weightfile_path(uid, d_name, make_dirs=True):
    weightfile = f"./user_models/{uid}/"
    if make_dirs and not os.path.isdir(weightfile):
        os.makedirs(weightfile)
    return os.path.join(weightfile, f'{d_name}_model.pth.tar')


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
        # self.thread.start()
        # self.next_time = 0

    def __set_interval(self):
        next_time = time.time() + self.interval
        while not self.stopEvent.wait(next_time - time.time()):
            next_time += self.interval
            self.action(self.socket_id)

    def start(self):
        print('start timer')
        self.thread.start()

    def cancel(self):
        print('stop timer')
        self.stopEvent.set()


class MyHTTPHandler(BaseHTTPRequestHandler):
    """
    ### MyHTTPHandler beschreibt den Umgang mit HTTP Requests
    """

    # http://donghao.org/2015/06/18/override-the-__init__-of-basehttprequesthandler-in-python/

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

    def do_GET(self):
        # by julian
        if "/getNodes" in self.path:
            print("GET /getNodes")
            ### POST Request Header ###
            self.send_response(200)
            # self.send_header('Content-type', 'application/json')
            # self.send_header('Access-Control-Allow-Origin', self.headers['origin'])
            self.end_headers()

            # get query from paths
            query = self.path.split('?')[1]
            # check for more than one param
            if "&" in self.path:
                return self.wfile.write("ERROR: API /getNodes does not allow multi params")
            # get file name from query and add .json ending
            fileName = f"{query.split('=')[1]}.json"

            file = os.path.join(DATA_DIR, fileName)
            print(f'Request file: {file}')

            with open(file, 'rb') as file:
                self.wfile.write(file.read())  # Read the file and send the contents

    def do_POST(self):
        """
        definiert den Umgang mit POST Requests
        Liest den Body aus - gibt ihn zum konvertieren weiter
        """
        global user_datas
        if self.path == "/nodes":
            try:
                print("post /nodes")
                ### POST Request Header ###
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                # self.send_header('Access-Control-Allow-Origin', self.headers['origin'])
                self.end_headers()

                # get body from request
                content_len = int(self.headers['Content-Length'])
                body = self.rfile.read(content_len)

                # convert body to list
                # data = json.loads(str(body).decode('utf-8'))  # python 2
                data = json.loads(str(body, encoding='utf-8'))  # python 3

                # DEBUGging 'data'
                # print "\033[32;1m", "DATA:", data, "\033[0m"
                print("  -> data:", [(k, type(v)) for (k, v) in data.items()])

                try:
                    user_id = data["userId"]
                except KeyError:
                    user_id = 0  # DEBUG
                    print("WARNING: No `userId` given, assigning 0.")

                # Katjas code goes here

                if "init" in data.keys():  # initial call
                    # choose and note dataset for user
                    dataset_name = data["dataset"]
                    if user_id not in user_datas:
                        user_datas[user_id] = UserData(user_id)
                    user_datas[user_id].dataset = dataset_name
                    # user_datas[user_id].idx_to_id = dict([(k, v['name']) for k, v in data['nodes'].iteritems()])
                    user_datas[user_id].svm_ids = np.array(
                        [v['index'] for v in data['nodes'].values()])  # store which keys are used, for the svm

                    # if dataset not yet initialized, catch up
                    if dataset_name not in initial_datas:
                        initialize_dataset(dataset_name)
                    initial_data = initial_datas[dataset_name]

                    # delete old model
                    weightfile = weightfile_path(user_id, dataset_name, make_dirs=False)
                    if data['init'] == 'new':
                        if os.path.isfile(weightfile):
                            print("DEBUG!! NEW: removing old model")
                            os.remove(weightfile)
                        else:
                            print("DEBUG!! NEW: no old model to remove")
                    else:
                        if os.path.isfile(weightfile):
                            print("DEBUG!! RESUME: keeping old model")
                        else:
                            print("DEBUG!! RESUME: no old model to keep")

                    # set attributes for katja legacy and svm training
                    if os.path.isfile(weightfile) and data['init'] == 'resume':
                        proj = generate_projections(weightfile, initial_data['features_raw'])
                        sendback = True
                        print("DEBUG! Generated Projections")
                    else:
                        proj = initial_data['projection']
                        sendback = False
                        print(f"DEBUG! Projections not generated. isfile: {os.path.isfile(weightfile)}, "
                              f"init: {data['init']}")

                    user_datas[user_id].graph_df = communication.make_graph_df(
                        image_ids=initial_data['image_id'],
                        projection=proj,
                        info_df=initial_data['info'],
                        coordinate_range=LIMITS)

                    graph_json = communication.graph_df_to_json(user_datas[user_id].graph_df, max_elements=MAX_DISPLAY)

                    user_datas[user_id].index_to_id = communication.make_index_to_id_dict(graph_json)
                    # embed() #DEBUG!!

                    # send projections on resume call
                    if sendback:
                        self.wfile.write(graph_json)
                    else:
                        self.wfile.write(b'{"done": true}')
                    return

                else:  # subsequent call
                    # finetuning... not working yet with dim 512, but 4096 needs new feature files
                    # load model
                    with torch.cuda.device(DEVICE):
                        dataset_name = user_datas[user_id].dataset
                        initial_data = initial_datas[dataset_name]

                        weightfile = weightfile_path(user_id, dataset_name)

                        if os.path.isfile(weightfile):  # not first iteration
                            model = load_model(weightfile)
                        else:  # first iteration
                            model = mapnet(N_LAYERS, pretrained=True, new_pretrain=True)
                        model.cuda()

                        """# gen labels [OLD]
                        lbl = [(int(k), v['groupId']) for k, v in data["nodes"].iteritems()]
                        lbl.sort(key=lambda x:x[0])
                        idx, lbl = zip(*lbl)
                        assert min(idx) == 0 and max(idx) == len(idx) - 1, "Not all nodes given in POST/nodes"
                        lbl = [x if x is not None else 0 for x in lbl]
                        lbl = np.array(lbl, dtype=np.long)
                        """
                        # gen labels sorting via image_id
                        # TODO: time data preparation
                        lbl_dict = {v['name']: v['groupId'] for v in data['nodes'].values()}
                        id_feat = zip(initial_data['image_id'], initial_data['features_raw'])
                        ids, labels, features = zip(
                            *[(name, lbl_dict[name], feat) for name, feat in id_feat if name in lbl_dict])
                        unique, labels = np.unique(labels,
                                                   return_inverse=True)  # works only in python 2.7, since None < all
                        labels = np.array(labels) if None in unique else np.array(labels) + 1  # ensure empty label is 0
                        features = np.array(features)
                        del lbl_dict, id_feat, unique
                        # TODO: end timing

                        train.train_mapnet(model, features, labels, verbose=True, outpath=weightfile)

                        # generate projection with new model
                        proj = generate_projections(model, features)

                        # reply with new projection
                        user_datas[user_id].graph_df = communication.make_graph_df(
                            image_ids=ids, projection=proj,
                            info_df=initial_data['info'],
                            coordinate_range=LIMITS)
                        graph_json = communication.graph_df_to_json(user_datas[user_id].graph_df,
                                                                    max_elements=MAX_DISPLAY)
                        user_datas[user_id].index_to_id = communication.make_index_to_id_dict(graph_json)

                        self.wfile.write(graph_json)
                        return
                """
                else: # no 'init' in `data`
                    print("ERROR: No init given")
                    self.wfile.write("{'error': 'no init-flag given'}")
                    return
                """

                # shortcut for data access
                # initial_data = initial_datas[user_datas[user_id].dataset]

                # print "\033[31;1m", "JSON:", graph_json, "\033[0m" #DEBUG!
                self.wfile.write(graph_json)  # body zurueckschicken
            except Exception as e:
                self.wfile.write(json.dumps({"error": {"msg": str(e),
                                                       "type": str(type(e)),
                                                       "loc": "/nodes"}
                                             }))  # error body zurueckschicken
                raise

        if self.path == "/getGroupNeighbours":
            try:
                print("post /getGroupNeighbours")
                ### POST Request Header ###
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # get body from request
                content_len = int(self.headers['Content-Length'])
                body = self.rfile.read(content_len)

                # convert body to list
                # data = json.loads(str(body).decode('utf-8'))  # python 2
                data = json.loads(str(body, encoding='utf-8'))  # python 3

                # DEBUG!
                print("\033[32;1m", "DATA:", data, "\033[0m")

                # choose right dataset
                user_id = data['userId']
                user_data = user_datas[user_id]
                dataset_name = user_data.dataset
                initial_data = initial_datas[dataset_name]

                group_id = int(data['groupId'])
                thresh = float(data['threshold'])
                if 'negatives' not in data.keys():  # first iteration
                    idx_pos = data['positives']
                    idx_neg = None
                    user_data.svm_negs = set()
                else:
                    idx_pos = data['positives']
                    idx_neg = (set(data['negatives']) | user_data.svm_negs).difference(idx_pos)
                    user_data.svm_negs = idx_neg

                feat = initial_data['features'][user_data.svm_ids]
                # convert idx_pos, idx_neg here
                idx_pos_inner = np.in1d(user_data.svm_ids, idx_pos).nonzero()[0]
                idx_neg_inner = np.in1d(user_data.svm_ids, idx_neg).nonzero()[0]
                neighbor_idcs, scores, svm = svm_k_nearest_neighbors(feat, idx_pos_inner, idx_neg_inner,
                                                                     max_rand_negatives=10,
                                                                     k=-1, verbose=False)
                neighbor_idcs = user_data.svm_ids[neighbor_idcs].tolist()

                """ #DEBUG!
                reversed_idcs = user_data.svm_ids[idx_pos_inner]
                print "REVERSED IDX (debug):", reversed_idcs
                """

                # make json
                return_dict = {'group': idx_pos,
                               'neighbours': dict(zip(neighbor_idcs, 1. - scores))}  # reverse scores
                return_dict = json.dumps(return_dict).encode()
                self.wfile.write(return_dict)  # body zurueckschicken
            except Exception as e:
                self.wfile.write(json.dumps({"error": {"msg": str(e),
                                                       "type": str(type(e)),
                                                       "loc": "/getGroupNeighbors"}
                                             }, indent=4).encode('utf-8'))  # error body zurueckschicken
                raise

        """
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
        """
        return


if __name__ == "__main__":
    # config
    HOST_NAME = "localhost"
    PORT_NUMBER = 8023
    print(SPLASH)
    try:
        http_server = HTTPServer((HOST_NAME, PORT_NUMBER), MyHTTPHandler)
        print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
        http_server.serve_forever()
    except KeyboardInterrupt:
        print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
    http_server.socket.close()
