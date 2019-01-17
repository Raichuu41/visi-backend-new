import numpy as np
import pandas as pd
import json

from aux import scale_to_range


def make_graph_df(image_ids, projection, info_df=None, coordinate_range=(-1, 1)):
    coordinates = scale_to_range(projection, coordinate_range[0], coordinate_range[1])
    df = pd.DataFrame(data=coordinates, index=image_ids, columns=('x', 'y'))
    df['group'] = None
    if info_df is not None:
        df = pd.concat([df, info_df], axis=1, join_axes=[df.index])     # add available information to images with existing coordinates
    return df


def graph_df_to_json(graph_df, max_elements=None, random_state=123):
    """Convert to format 'number_index': {'index': number_index, 'name': id, 'x': x, 'y': y, 'labels': [l1, ..., ln]}"""
    if max_elements is not None and max_elements < len(graph_df):
        graph_df = graph_df.sample(max_elements, random_state=random_state)

    categories = sorted(c for c in graph_df.columns if c not in ['x', 'y'])
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
    return json.dumps(data).encode()


def make_index_to_id_dict(json_graph):
    graph = json.loads(json_graph)['nodes']
    index_id = map(lambda x: (x['index'], x['name']), graph.values())
    return dict(index_id)

