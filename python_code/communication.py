import numpy as np
import pandas as pd
import json
import warnings
import requests


# from aux import scale_to_range


def make_graph_df(image_ids, projection, info_df=None, coordinate_range=(-1, 1)):
    x = scale_to_range(projection[:, 0], coordinate_range[0], coordinate_range[1])
    y = scale_to_range(projection[:, 1], coordinate_range[0], coordinate_range[1])
    coordinates = np.stack([x, y], axis=1)
    df = pd.DataFrame(data=coordinates, index=image_ids, columns=('x', 'y'))
    df['group'] = None
    df['weight'] = None
    if info_df is not None:
        df = pd.concat([df, info_df], axis=1, join_axes=[df.index])     # add available information to images with existing coordinates
    return df


def graph_df_to_json(graph_df, max_elements=None, display_indices=None, random_state=123, socket_id=None):
    """Convert to format 'number_index': {'index': number_index, 'name': id, 'x': x, 'y': y, 'labels': [l1, ..., ln]}"""
    if max_elements is not None and max_elements < len(graph_df):
        if display_indices is not None:
            display_df = graph_df.loc[display_indices]
            if len(display_indices) > max_elements:
                print('Cannot display all given indices () because max_elements is {}.'
                      .format(len(display_indices), max_elements))
                graph_df = display_df.sample(max_elements, random_state=random_state)
            else:
                rest_idcs = np.setdiff1d(graph_df.index.values, display_df)
                n_missing = max_elements - len(display_indices)
                rest_df = graph_df.loc[rest_idcs].sample(n_missing, random_state=random_state)
                graph_df = graph_df.append(rest_df)
        else:
            graph_df = graph_df.sample(max_elements, random_state=random_state)

    categories = sorted(c for c in graph_df.columns if c not in ['x', 'y', 'group', 'weight'])
    label_values = graph_df[categories].values

    def merge_label_columns(rows):
        return([None if pd.isnull(v) else v for v in rows])

    labels = map(merge_label_columns, label_values)

    data = np.stack([range(len(graph_df)), graph_df.index.values, graph_df['x'].values, graph_df['y'].values], axis=1)
    converted_df = pd.DataFrame(columns=['index', 'name', 'x', 'y'],
                                data=data)
    converted_df['labels'] = labels

    converted_graph = converted_df.to_dict(orient='index')

    # encode to json
    data = {'nodes': converted_graph, 'categories': categories}
    if socket_id is not None:
        data['socket_id'] = socket_id
    return json.dumps(data).encode()


def json_graph_to_df(json_graph):
    graph = json_graph if isinstance(json_graph, dict) else json.loads(json_graph)['nodes']
    return pd.DataFrame.from_dict(graph, orient='index')


def make_index_to_id_dict(json_graph):
    graph = json_graph if isinstance(json_graph, dict) else json.loads(json_graph)['nodes']
    index_id = map(lambda x: (x['index'], x['name']), graph.values())
    return dict(index_id)


def update_labels_and_weights(old_labels, old_weights, new_labels, new_weights, threshold=0):
    """Function to update label and weight.
    old_label      new_label
        p              p           replace by new label if its weight is larger
        p              n           replace by negative if old weight is not 1 (i.e. if old label was a prediction)
        n              p           replace only if new weight is 1 (i.e. new label is not a prediction but was labeled)
        n              n           replace with a chance of 50%
    """
    def map_fn(old_label, old_weight, new_label, new_weight):
        if not (new_label is None or new_label == 0) and new_weight < 0:
            raise RuntimeError('New weight must not be smaller than zero.')

        if new_weight < threshold or new_label == 0:       # leave unchanged
            return old_label, old_weight

        elif old_label is None or old_label == 0:
            return new_label, new_weight

        elif old_label > 0 and new_label > 0:
            if new_weight >= old_weight:
                if old_weight == 1 and old_label != new_label:
                    warnings.warn('Overwrite existing label, even though its weight was 1.')
                return new_label, new_weight
            else:
                return old_label, old_weight

        elif old_label > 0 and new_label < 0:
            if old_weight != 1:
                return new_label, new_weight
            elif old_label == -new_label:       # sample was labeled false before
                return new_label, new_weight
            else:
                return old_label, old_weight

        elif old_label < 0 and new_label > 0:
            if new_weight == 1:
                return new_label, new_weight
            else:
                return old_label, old_weight

        elif old_label < 0 and new_label < 0:
            replace = np.random.choice([0, 1])
            if replace:
                return new_label, new_weight
            else:
                return old_label, old_weight
        else:
            raise RuntimeError('Unknown label case. Have old_label={}, old_weight={}, new_label={}, new_weight={}.')
    test_lambda = [old_labels, old_weights, new_labels, new_weights]
    labels, weights = np.stack(map(lambda test_lambda:
                                   map_fn(old_label=test_lambda[0], old_weight=test_lambda[1], new_label=test_lambda[2], new_weight=test_lambda[3]),
                                   zip(old_labels, old_weights, new_labels, new_weights)), axis=1)
    return labels, weights


def send_payload(payload):
    headers = {'content-type': 'application/json'}
    requests.post("http://localhost:3000/api/v1/updateEmbedding", data=payload, headers=headers)
