import os
import deepdish as dd

feature_file = 'features/MobileNetV4_info_artist_49_multilabel_test_full_images_128.hdf5'
os.chdir('./MapNetCode/pretraining')

def load_feature(feature_file):
    assert os.path.isfile(feature_file), 'Feature file not found. Extraction not yet implemented.'

    data = dd.io.load(feature_file)