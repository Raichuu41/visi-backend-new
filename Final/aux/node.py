"""This file contains all the functions and classes related to creating and interacting with nodes.
For collections of nodes pandas.DataFrame is used as container.
Nodes have to be a dictionary where each key corresponds to the name/ID of the node
and the value is a dictionary with the attributes ."""
import pandas as pd


def nodes_to_df(nodes):
    columns = sorted(nodes[0].value.keys())       # ensure columns are sorted
    return pd.DataFrame.from_dict(nodes, orient='index', columns=columns)


def df_to_nodes(df):
    return df.to_dict(orient='index')





