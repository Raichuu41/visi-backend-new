# copy files from class folders into one folder

import os
from shutil import copy


image_dir = '/export/home/kschwarz/Documents/Data/CUB_200_2011/images_nofolders'
outdir = '/export/home/kschwarz/Documents/Masters/ManifoldMetric/rmac-master/datasets/CUB_train'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

for root, dirs, files in os.walk(image_dir):
    print('copy files from folder {}'.format(root))
    for f in files:
        copy(os.path.join(root, f), outdir)
