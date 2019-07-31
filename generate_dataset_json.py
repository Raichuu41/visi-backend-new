#!/usr/bin/env python

import os
import json
from argparse import ArgumentParser
from python_code.initialization import Initializer

DEFAULT_DATA_PATH = "../visiexp/datasets/"
FILE_EXTENTIONS = ["jpg", "png"]
FEATURE_DIM = 512

EXAMPLES = """$ python generate_dataset_json.py -n Demo_Wiki_1 -i demo_wiki_1 -r ~/visiexp/datasets -a
 -> Generates a dataset called 'Demo_Wiki_1' with all images in '~/visiexp/datasets/raw/demo_wiki_1/imgs'

 $ python generate_dataset_json.py -n Demo_Wiki_2 -i demo_wiki_1 -r ~/visiexp/datasets -f files.txt
 -> Generates a dataset called 'Demo_Wiki_2' with all images listed in 'files.txt' located in '~/visiexp/datasets/raw/demo_wiki_1/imgs'
 
 $ python generate_dataset_json.py -n Demo_Wiki_3 -i demo_wiki_1 -r ~/visiexp/datasets -e bmp -a
 -> Generates a dataset called 'Demo_Wiki_3', using only bmp-files"""

parser = ArgumentParser(description="Generates server-readable JSON files for new datasets\n" + EXAMPLES)
parser.add_argument("--root", "-r", type=str, default=DEFAULT_DATA_PATH, help="Root directory of all datasets.")
parser.add_argument("--name", "-n", type=str, required=True, help="Name of the new dataset.")
parser.add_argument("--idir", "-i", type=str, required=True, help="Name of image folder inside `<root>/raw/`.")
parser.add_argument("--extentions", "-e", type=str, nargs="+", default=FILE_EXTENTIONS, help="Possible image extentions. Default: {}".format(FILE_EXTENTIONS))
parser_input = parser.add_mutually_exclusive_group(required=True)
parser_input.add_argument("--data", "-d", type=str, nargs="+", help="List of names of all data images to be processed. Preceding paths are ignored for easier auto-completion. Conflicts with `-f`, `-a`.")
parser_input.add_argument("--file", "-f", type=str, nargs="?", help="File that contains the names of all data images to be processed. Conflicts with `-d`, `-a`.")
parser_input.add_argument("--all", "-a", action="store_true", help="Use all files in image folder. Conflicts with `-d`, `-f`.")
parser.add_argument("--silent", "-s", action="store_true", help="Don't be verbose.")

def probe_image_file(name, idir):
    if any(name.endswith("." + ext) for ext in args.extentions):
        # if extention is given, only test given extention
        if os.path.isfile(os.path.join(idir, name)):
            return name
    else:
        # else, try all possible extention
        for ext in args.extentions:
            tmp = os.path.join(idir, name + "." + ext)
            if os.path.isfile(tmp):
                return name
    return None 

if __name__ == "__main__":
    args = parser.parse_args()

    json_path = os.path.join(args.root, args.name + ".json")
    if os.path.exists(json_path):
        raise IOError("JSON for `{}` already exists.".format(args.name))

    imdir = os.path.join(args.root, "raw", args.idir, "imgs")
    if not os.path.isdir(imdir):
        raise IOError("The path `{}` does not exist or is not a directory.".format(imdir))

    # check input data
    if not args.silent: print "### Checking Files ###"
    if args.data is not None:
        data = [os.path.split(x)[1] for x in args.data]
    elif args.file is not None:
        data = [x.strip() for x in open(args.file, 'r')]
    elif args.all is True:
        data = os.listdir(imdir)
    else:
        assert False, "This should not be happening!"
    files = [probe_image_file(f, imdir) for f in data]
    if None in files:
        fail = "; ".join([data[i] for i,b in enumerate(files) if b is None])
        raise IOError("The following data could not be found: {}".format(fail))

    # generate temporary json to call Initializer
    if not args.silent: print "### Generating temporary JSON ###"
    nodes = {name:{'label':'', 'idx':i, 'x':0, 'y':0} for i, name in enumerate(data)}
    out = {'im_dir_name': args.idir, 'nodes': nodes, 'temporary': True}
    json.dump(out, open(json_path, "w"))

    # generate additional resources with Initializer
    if not args.silent: print "### Loading Initializer ###"
    dot_extentions = ["." + e for e in args.extentions] + [""]
    init = Initializer(args.name, impath=imdir, info_file=json_path, outdir=args.root, feature_dim=512, data_extensions=dot_extentions, verbose=True)
    init.initialize()
    proj = init.make_projection_dict()

    # store real JSON with projections
    if not args.silent: print "### Generating final JSON ###"
    for i in range(len(proj['image_id'])):
        for ext in dot_extentions:
            try:
                nodes[proj['image_id'][i] + ext]['x'] = proj['projection'][i][0]
                nodes[proj['image_id'][i] + ext]['y'] = proj['projection'][i][1]
            except KeyError:
                pass
            else: # break, if key worked
                break
        else:
            raise IOError("No extention worked for {}".format(proj['image_id'][i])) # (this should not be happening)
    out = {'im_dir_name': args.idir, 'nodes': nodes}
    json.dump(out, open(json_path, "w"))