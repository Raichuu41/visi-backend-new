#!/usr/bin/env python
import os
import json
from argparse import ArgumentParser
from python_code.initialization import Initializer
import torch
import numpy as np
from server import FEATURE_DIM

DEFAULT_DATA_PATH = "../../images/"
FILE_EXTENTIONS = ["jpg", "png"]

README = """\
1. copy all images into <datasetroot>/raw/<SOMENAME>/imgs/
   (please don't use multiple extentions [e.g. 'image.png.jpg'], this has not been tested due to time)
2. give a+rx rights recursively (`chmod a+rx raw/<SOMENAME>`)
3. run this script with a list (subset) of images or use all images in the folder [See EXAMPLES]
4. punch julian to add the script in the node.js-server
"""

EXAMPLES = """$ python generate_dataset_json.py -n Demo_Wiki_1 -i demo_wiki_1 -r ~/visiexp/datasets -a
 -> Generates a dataset called 'Demo_Wiki_1' with all images in '~/visiexp/datasets/raw/demo_wiki_1/imgs'

 $ python generate_dataset_json.py -n Demo_Wiki_2 -i demo_wiki_1 -r ~/visiexp/datasets -f files.txt
 -> Generates a dataset called 'Demo_Wiki_2' with all images listed in 'files.txt' located in '~/visiexp/datasets/raw/demo_wiki_1/imgs'
 
 $ python generate_dataset_json.py -n Demo_Wiki_3 -i demo_wiki_1 -r ~/visiexp/datasets -e bmp -a
 -> Generates a dataset called 'Demo_Wiki_3', using only bmp-files"""

LABELFILE_EXAMPLE = """\
CATEGORIES:         animal       pose     color
some_wild_bear      bear         standing brown
cat_25_06_2003      cat          sitting  white
IMG_20190229_235901 wolpertinger flying   gray
# first line defines categories ("CATEGORIES:" mandatory!)
# use as many spaces as you please
# non-mentioned images won't receive lables
"""

parser = ArgumentParser(description="Generates server-readable JSON files for new datasets")
parser.add_argument("--root", "-r", type=str, default=DEFAULT_DATA_PATH, help="Root directory of all datasets.")
parser.add_argument("--name", "-n", type=str, required=True, help="Name of the new dataset.")
parser.add_argument("--idir", "-i", type=str, required=True, help="Name of image folder inside `<ROOT>/raw/`.")
parser.add_argument("--lfile", "-l", type=str, help="File that contains labels in categories.")
parser.add_argument("--extentions", "-e", type=str, nargs="+", default=FILE_EXTENTIONS, help="Possible image extentions. Default: {}".format(FILE_EXTENTIONS))
parser_input = parser.add_mutually_exclusive_group(required=True)
parser_input.add_argument("--data", "-d", type=str, nargs="+", help="List of names of all data images to be processed. Preceding paths are ignored for easier auto-completion. Conflicts with `-f`, `-a`.")
parser_input.add_argument("--file", "-f", type=str, nargs="?", help="File that contains the names of all data images to be processed. Currently only method to add labels. Conflicts with `-d`, `-a`.")
parser_input.add_argument("--all", "-a", action="store_true", help="Use all files in image folder. Conflicts with `-d`, `-f`.")
parser.add_argument("--silent", "-s", action="store_true", help="Don't be verbose.")
parser.add_argument("--device", "-x", type=int, default=0, help="CUDA device to use")

def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))

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

def clean_exts(inp_l):
    res = []
    for item in inp_l:
        if any(item.endswith("." + ext) for ext in args.extentions):
            res.append(item.rsplit(".", 1)[0])
        else:
            res.append(item)
    return res

if __name__ == "__main__":
    args = parser.parse_args()

    # with torch.cuda.device(args.device):
    if True:
        json_path = os.path.abspath(os.path.join(__file__, DEFAULT_DATA_PATH, 'init_json', f'{args.name}.json'))
        if os.path.exists(json_path):
            raise IOError("JSON for `{}` already exists.".format(args.name))

        imdir = os.path.abspath(os.path.join(__file__, f'{args.root}{args.idir}'))
        if not os.path.isdir(imdir):
            raise IOError("The path `{}` does not exist or is not a directory.".format(imdir))

        # check input data
        if not args.silent: print("### Checking Files ###")
        if args.data is not None:
            data = [os.path.split(x)[1] for x in args.data]
        elif args.file is not None:
            data = [x.strip() for x in open(args.file, 'r')]
        elif args.all is True:
            data = os.listdir(imdir)
        else:
            assert False, "This should not be happening!"
        data = sorted(data)
        # error, if files missing
        files = [probe_image_file(f, imdir) for f in data]
        if None in files:
            fail = "; ".join([data[i] for i,b in enumerate(files) if b is None])
            raise IOError("The following data could not be found: {}".format(fail))
        data = clean_exts(data)

        # generate nodes-part for json
        nodes = {name:{'label': '', 'labels': [], 'idx':i, 'x':0, 'y':0} for i, name in enumerate(data)}
        #               ^-- this is needed by the Initializer (probably legacy)

        # assign labels, if given
        if args.lfile is not None:
            if not args.silent: print("### Assigning Labels ###")
            if not os.path.isfile(args.lfile):
                raise IOError("The label file could not be found: {}".format(args.lfile))
            labelfile = open(args.lfile, 'r')
            first_line = labelfile.readline().split()
            if first_line[0] != "CATEGORIES:":
                raise IOError("The label file has wrong formatting.")
            if not first_line[1:]:
                raise IOError("The label file has no categories.")
            categories = first_line[1:]
            for i, line in enumerate(labelfile, 1):
                line = line.split()
                if len(line) != len(categories) + 1:
                    raise IOError("The label file has the wrong amount of labels in line {}".format(i))
                img, lables = line[0], line[1:]
                if img not in nodes:
                    raise IOError("Unknown image id in label file: {}".format(img))
                nodes[img]['labels'] = lables
        else:
            categories = None

        try:
            # generate temporary json to call Initializer
            if not args.silent: print("### Generating temporary JSON ###")
            out = {'im_dir_name': args.idir, 'nodes': nodes, 'temporary': True}
            json.dump(out, open(os.path.abspath(os.path.join(__file__, f'{args.root}{args.name}.json')), "w"))

            # generate additional resources with Initializer
            if not args.silent: print("### Loading Initializer ###")
            dot_extentions = ["." + e for e in args.extentions] + [""]
            init = Initializer(args.name, impath=imdir, info_file=json_path, outdir=args.root, feature_dim=FEATURE_DIM,
                               data_extensions=dot_extentions, verbose=True)
            init.initialize(raw_features=True)
            proj = init.make_projection_dict()

            # store real JSON with projections
            if not args.silent: print("### Generating final JSON ###")
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
            if categories is not None:
                out['categories'] = categories
        except:
            if not args.silent: print("!#! An error occured, deleting temp files !#!")
            if os.path.isfile(json_path):
                os.remove(json_path)
            path = os.path.join(args.root, "norm", "{}_mean_std.pkl")
            if os.path.isfile(path):
                os.remove(path)
            path = os.path.join(args.root, "features", "{}_512.h5")
            if os.path.isfile(path):
                os.remove(path)
            path = os.path.join(args.root, "features", "{}_512_PCA.h5")
            if os.path.isfile(path):
                os.remove(path)
            raise
        # sort the node keys to be equal to the sorted image list
        out['nodes'] = sorted(out['nodes'].items())
        out['nodes'] = {x[0]: x[1] for x in out['nodes']}
        json.dump(out, open(json_path, "w"), default=default, indent=4)
