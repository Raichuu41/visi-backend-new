#!/usr/bin/env python

import os
import json
from argparse import ArgumentParser

import deepdish as dd

parser = ArgumentParser(description="Converts Katjas HDF5-Files for datasets to JSON-Files")
parser.add_argument('--input', default='./dataset_info', type=str)
parser.add_argument('--proj_dir', default='./initial_projections', type=str)
parser.add_argument('--feat_dir', default='./features', type=str)
parser.add_argument('--output', default='./json', type=str)
parser.add_argument('--ask', action="store_true")


args = parser.parse_args()

def create_json(fname, proj, ask):
    d = dd.io.load(fname)["df"]
    if 'df' in d:
        d = d['df'] # I'm a DataFrame now \o/
    index = d.keys()[0]
    d = d.to_dict()
    d["nodes"] = dict()
    
    for i, (key, label) in enumerate(d[index].iteritems()):
        d["nodes"][key] = {"label": label}
    del d[index]

    if proj:
        proj = dd.io.load(proj)
        for i, key in enumerate(proj['image_id']):
            x, y = proj["projection"][i].astype(float)
            d["nodes"][key].update({"x": x, "y": y, "idx": i})
    
    if ask:
        impath = raw_input("Name of Image folder:")
        d["im_dir_name"] = impath

    return d

if os.path.isdir(args.input):
    for fname in os.listdir(args.input):
        fpath = os.path.join(args.input, fname)
        projpath = os.path.join(args.proj_dir, fname[5:])
        # featpath = os.path.join(args.feat_dir, fname[5:-3] + "_512.h5")
        if os.path.isfile(fpath) and fname.startswith("info_"):
            print "Processing {}...".format(fname)
            if os.path.isfile(projpath):
                d = create_json(fpath, projpath, args.ask)
            else:
                print "Skipping positions from {}".format(projpath)
                d = create_json(fpath, None, args.ask)
            outpath = os.path.join(args.output, fname[5:-2] + "json")
            json.dump(d, open(outpath, "w"))
        else:
            print "Skipping {}...".format(fname)
            continue

# elif os.path.isfile(args.input):
#         create_json(args.input)

else:
    print "`input` is not a valid file or directory"