import pickle
import sys
sys.path.append('../FullPipeline')
from aux import imgrid
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


with open('_svm_prediction.pkl', 'rb') as f:
    svm_data = pickle.load(f)
    image_names = svm_data['image_names']
    distances = svm_data['distance']
    local_idcs = svm_data['local_indices']

image_names = image_names[local_idcs]
distances = distances[local_idcs]

sorted_images = np.array(sorted([(d, n) for d, n in zip(distances, image_names)]))

# load image
def load_img(impath, name, size=(100,100)):
    return Image.open(os.path.join(impath, name)).resize(size)

impath = '/export/home/kschwarz/Documents/Masters/WebInterface/images'
# idcs = range(0, len(image_names), 2)
idcs = range(len(image_names))
images = [load_img(impath, n[1] + '.jpg') for n in sorted_images[idcs]]

fig = plt.figure(figsize=(18, 12))
imgrid(images, ncols=18, fig=fig)