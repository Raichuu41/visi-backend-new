# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:25:24 2018
@author: jlibor
"""
# Aktuelle Version als Hilfe ausgeben
import os
import sys
import numpy as np
import torch
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
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import mysql.connector

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
        self.dataset = ''  # dataset the user is working on
        self.svm_ids = []
        self.svm_negs = set()

        # LEGACY: remove asap
        self.index_to_id = None
        self._svm_temp = None
        self.graph_df = None

    def __repr__(self):
        return f"<UserData({self.user_id}) at {id(self)}>"

    def get_user_id(self):
        return self.user_id

    def get_current_dataset(self):
        return self.dataset

def initialize_dataset(dataset_name):
    print(f'Initialize {dataset_name}...')
    data_info_file = os.path.join(DATA_DIR, f'{dataset_name}.json')
    initializer = init.Initializer(dataset_name, impath=IMPATH, info_file=data_info_file,
                                   feature_dim=FEATURE_DIM, outdir=DATA_DIR)
    initializer.initialize(dataset=IMPATH is not None, is_test=dataset_name.endswith('_test'), raw_features=True)
    initial_datas[dataset_name] = initializer.get_data_dict(normalize_features=True)
    # sort the features, projection and image IDs
    sorted_idx = np.argsort(initial_datas[dataset_name]['image_id'])
    initial_datas[dataset_name]['image_id'] = [initial_datas[dataset_name]['image_id'][i] for i in sorted_idx]
    initial_datas[dataset_name]['info'] = initial_datas[dataset_name]['info'].iloc[sorted_idx]
    for key, value in initial_datas[dataset_name].items():
        if isinstance(initial_datas[dataset_name][key], np.ndarray):
            initial_datas[dataset_name][key] = value[sorted_idx]
    print(f'Done [{dataset_name}].')


def generate_projections(model, features):
    if type(model) is str:
        model = load_model(model)
    model.eval()
    if nvidia_gpu_available:
        model.cuda()
    ds = torch.utils.data.TensorDataset(torch.tensor(features))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    proj = []
    for item, *_ in dl:
        if nvidia_gpu_available:
            item = item.cuda()
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
    weightfile = f"./user_models/user_id_{uid}/"
    if make_dirs and not os.path.isdir(weightfile):
        os.makedirs(weightfile)
    return os.path.join(weightfile, f'{d_name}_model.pth.tar')


def save_snapshot(data, snapshot_id):
    with open(f'snapshots/snapshot_{snapshot_id}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_model(model, snapshot_id):
    model_path = f'snapshots/model_{snapshot_id}.pth'
    torch.save(model.state_dict(), model_path)


def delete_due_models():
    """
    | Deletes temporary models which exist for too long where we can assume it is out of use.
    | Can only happen if the user didn't close the application properly
    """
    current_time = time.perf_counter()
    max_duration = 86400  # one day in seconds
    to_delete = [user_id for user_id, (_, timestamp) in temporary_models.items()
                 if current_time - timestamp >= max_duration]
    for user_id in to_delete:
        temporary_models.pop(user_id, None)
        print(f'Deleted outdated model of user {user_id}!')


def delete_snapshot(snapshot_id):
    snapshot_path = f'snapshots/snapshot_{snapshot_id}.json'
    model_path = f'snapshots/model_{snapshot_id}.json'
    if os.path.exists(model_path):
        os.remove(model_path)  # delete old training model if available
        print(f'Successfully deleted the training model with ID {snapshot_id}!')
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
        print(f'Successfully deleted the snapshot with ID {snapshot_id}!')
    else:
        raise ValueError(f'Something went wrong. There was no snapshot with ID {snapshot_id}!')


def check_amount_of_snapshots(user_id, dataset_id, max_allowed=2):
    """
    | Checks the amount of snapshots saved in the database for given user ID and dataset ID.
    | If the amount exceeds the maximum allowed, the oldest snapshots will be deleted

    :param int user_id: ID of the user
    :param int dataset_id: ID of the dataset
    :param int max_allowed: Maximum allowed number of saved snapshots for a given user on the specified dataset
    """
    query = '''select * from visiexp.snapshots where user_id = %s and dataset_id = %s'''
    cursor = db_connection.cursor()
    cursor.execute(query, (user_id, dataset_id))
    results = cursor.fetchall()
    results_amount = len(results)
    if results_amount > max_allowed:
        # figure out how many snapshots have to be deleted
        amount_to_delete = results_amount - max_allowed
        # get the snapshot IDs which will be deleted for later usage
        get_ids_query = '''select snapshot_id from visiexp.snapshots where user_id = %s and dataset_id = %s
                        order by created_at limit %s'''
        cursor.execute(get_ids_query, (user_id, dataset_id, amount_to_delete))
        deleted_snapshot_ids = [snapshot_id[0] for snapshot_id in cursor.fetchall()]
        # delete the oldest snapshots
        delete_query = '''delete from visiexp.snapshots where user_id = %s and dataset_id = %s
                        order by created_at limit %s'''
        cursor.execute(delete_query, (user_id, dataset_id, amount_to_delete))
        # delete the data of snapshots which were deleted from the database
        for snapshot_id in deleted_snapshot_ids:
            delete_snapshot(snapshot_id)


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
        if '/checkTemporaryModels' in self.path:
            self.send_response(200)
            self.end_headers()
            delete_due_models()
            self.wfile.write(b'{}')

        if "/getNodes" in self.path:
            print("GET /getNodes")
            # POST Request Header
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
        if '/getSnapshots' in self.path:
            self.send_response(200)
            self.end_headers()
            query = self.path.split('?')[1]
            # extract the parameters, expected format: abc=123&xyz=456
            user_id, dataset_id = [x.split('=')[-1] for x in query.split('&')]
            cursor = db_connection.cursor(dictionary=True)
            query = '''select * from visiexp.snapshots where user_id = %s and dataset_id = %s'''
            cursor.execute(query, (user_id, dataset_id))
            result = cursor.fetchall()
            self.wfile.write(json.dumps(result, indent=4, default=str).encode())
        if '/resetTempModel' in self.path:
            self.send_response(200)
            self.end_headers()
            query = self.path.split('?')[1]
            user_id = int(query.split('=')[1])
            temporary_models.pop(user_id, None)  # delete previous temporary model if available
            print(f'Reset temporary model of user ID {user_id}')
            self.wfile.write(b'{}')

        if '/loadSnapshot' in self.path:
            self.send_response(200)
            self.end_headers()
            query = self.path.split('?')[1]
            first_part, second_part = query.split('&', 1)
            snapshot_id = first_part.split('=')[1]
            user_id = int(second_part.split('=')[1])
            with open(f'snapshots/snapshot_{snapshot_id}.json', 'r', encoding='utf-8') as snap_file:
                data = json.load(snap_file)
            if data['modelChanged']:  # load model if it was modified
                model = mapnet(N_LAYERS, pretrained=False, new_pretrain=True)
                model.load_state_dict(torch.load(f'snapshots/model_{snapshot_id}.pth'))
                model.eval()
                temporary_models[user_id] = (model, time.perf_counter())  # save model for future references
            self.wfile.write(json.dumps(data, indent=4, ).encode())

    def do_POST(self):
        """
        definiert den Umgang mit POST Requests
        Liest den Body aus - gibt ihn zum konvertieren weiter
        """
        global user_datas
        if '/saveSnapshot' in self.path:
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            content_len = int(self.headers['Content-Length'])
            body = self.rfile.read(content_len)
            data = json.loads(str(body, encoding='utf-8'))
            user_id = int(data['userid'])
            dataset_id = data['dataset']
            dataset_count = data['count']
            groups_count = len(data['groups'])
            snapshot_name = data['snapshotName']
            modified_model = data['modelChanged']
            display_count = data['displayCount']
            cursor = db_connection.cursor()
            # insert new row for snapshots data
            try:
                cursor.execute('''INSERT INTO visiexp.snapshots (user_id, snapshot_name, dataset_id, 
                modified_model, count, groups_count, display_count) VALUES (%s, %s, %s, %s, %s, %s, %s)''',
                               (user_id, snapshot_name, dataset_id, modified_model, dataset_count, groups_count,
                                display_count))
            except mysql.connector.Error as e:
                print(f'Error: {e}')
                self.send_response(500)
                self.wfile.write(b'{"status": "failed"}')
                return
            generated_snapshot_id = cursor.lastrowid
            check_amount_of_snapshots(user_id, dataset_id)
            save_snapshot(data, generated_snapshot_id)
            if modified_model:  # model was changed
                save_model(temporary_models[user_id][0], generated_snapshot_id)
            db_connection.commit()  # save changes to the database
            print(f'Successfully saved the snapshot {snapshot_name} with ID {generated_snapshot_id}!')
            self.wfile.write(b'{}')
        if self.path == "/nodes":
            try:
                print("post /nodes")
                # POST Request Header
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
                    user_id = int(data["userId"])
                except KeyError:
                    user_id = 0  # DEBUG
                    print("WARNING: No `userId` given, assigning 0.")

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
                        # this might not be necessary anymore (resets are done on switch dataset/logout)
                        temporary_models.pop(user_id, None)  # delete previous temporary model if available
                    # set attributes for katja legacy and svm training
                    if os.path.isfile(weightfile) and data['init'] == 'resume':
                        # create a fresh new model
                        temporary_models[user_id] = (mapnet(N_LAYERS, pretrained=False, new_pretrain=True),
                                                     time.perf_counter())
                        proj = generate_projections(temporary_models[user_id][0], initial_data['features_raw'])
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
                    # send projections on resume call
                    if sendback:
                        self.wfile.write(graph_json)
                    else:
                        self.wfile.write(b'{"done": true}')
                    return

                else:  # subsequent call
                    # finetuning... not working yet with dim 512, but 4096 needs new feature files
                    # load model
                    # with torch.cuda.device(DEVICE):
                    if True:
                        dataset_name = user_datas[user_id].get_current_dataset()
                        initial_data = initial_datas[dataset_name]
                        model = temporary_models[user_id][0] if user_id in temporary_models else mapnet(
                            N_LAYERS, pretrained=False, new_pretrain=True
                        )
                        if nvidia_gpu_available:
                            model.cuda()  # NVIDIA GPU is being used
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
                        # todo:: save temporary model to variable + save in snapshot when initiated
                        # save newly trained model temporarily (for later snapshot save)
                        temporary_models[user_id] = (train.train_mapnet(model, features, labels, verbose=True,
                                                                        use_gpu=nvidia_gpu_available),
                                                     time.perf_counter())
                        # generate projection with new model
                        proj = generate_projections(temporary_models[user_id][0], features)

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
                    user_data.svm_negs = []
                else:
                    idx_pos = data['positives']
                    idx_neg = [x for x in list(dict.fromkeys(data['negatives'] + user_data.svm_negs))
                               if x not in idx_pos]
                    # idx_neg = list((set(data['negatives']) | set(user_data.svm_negs)).difference(idx_pos))
                    user_data.svm_negs = idx_neg

                feat = initial_data['features'][user_data.svm_ids]

                # convert idx_pos, idx_neg here
                idx_pos_inner = np.in1d(user_data.svm_ids, idx_pos).nonzero()[0]
                idx_neg_inner = np.in1d(user_data.svm_ids, idx_neg).nonzero()[0]
                neighbor_idcs, scores, svm = svm_k_nearest_neighbors(feat, idx_pos_inner, idx_neg_inner,
                                                                     max_rand_negatives=10,
                                                                     k=-1, verbose=False)
                neighbor_idcs = user_data.svm_ids[neighbor_idcs].tolist()

                # make json
                return_dict = {'group': idx_pos,
                               'neighbours': dict(zip(neighbor_idcs, 1. - scores))}  # reverse scores
                max_score = max([x for x in return_dict['neighbours'].values()])
                return_dict = json.dumps(return_dict).encode()

                self.wfile.write(return_dict)  # body zurueckschicken
            except Exception as e:
                self.wfile.write(json.dumps({"error": {"msg": str(e),
                                                       "type": str(type(e)),
                                                       "loc": "/getGroupNeighbors"}
                                             }, indent=4).encode('utf-8'))  # error body zurueckschicken
                raise
        return


if __name__ == "__main__":
    # config
    HOST_NAME = "localhost"
    PORT_NUMBER = 8023
    print(SPLASH)
    """
    db_config = {
        'user': 'visi',
        'password': 'test123',
        'host': 'localhost',
        'database': 'visiexp',
        'raise_on_warnings': True
    }
    """
    db_config = {
        'user': 'visi',
        'password': '!vsearch12',
        'host': '129.206.106.202',
        'database': 'visiexp',
        'raise_on_warnings': True
    }
    temporary_models = {}  # expected key = user ID, value = tuple(model, timestamp)
    nvidia_gpu_available = torch.cuda.is_available()
    try:
        db_connection = mysql.connector.connect(**db_config)
        print('Successfully connected to the database!')
    except mysql.connector.Error:
        print('Failed to connect to the database!')
        exit()
    try:
        http_server = HTTPServer((HOST_NAME, PORT_NUMBER), MyHTTPHandler)
        print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
        http_server.serve_forever()
    except KeyboardInterrupt:
        print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER), '- Beenden mit STRG+C')
        http_server.socket.close()
