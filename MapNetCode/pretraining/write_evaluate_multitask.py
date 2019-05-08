import os
import math


feature_files_train = [f for f in os.listdir('./evaluation/pretrained_features') if 'subset_val' in f]
feature_files_train.sort()

feature_files_val = [f for f in os.listdir('./evaluation/pretrained_features') if 'subset_test' in f]
feature_files_val.sort()

n_files = 4
n_per_file = int(math.ceil(len(feature_files_train)*1.0 / n_files))

filecounter = 0

for filecounter in range(n_files):
    idx_start = filecounter * n_per_file
    idx_stop = min(len(feature_files_train), idx_start+n_per_file)
    with open('./pretraining/multitask_evaluation_{}.sh'.format(filecounter), 'w') as f:
        for ftrain, fval in zip(feature_files_train[idx_start:idx_stop],
                                feature_files_val[idx_start:idx_stop]):
            settings = 'python multitask_evaluation.py ' \
                       '--info_train ./pretraining/wikiart_datasets/info_elgammal_subset_val.hdf5 ' \
                       '--info_val ./pretraining/wikiart_datasets/info_elgammal_subset_test.hdf5 ' \
                       '--feature_train ./evaluation/pretrained_features/{} ' \
                       '--feature_val ./evaluation/pretrained_features/{} ' \
                       '--tasks artist_name,genre,style'.format(ftrain, fval)
            f.write(settings + '\n\n')

