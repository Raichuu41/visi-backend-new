import os
import deepdish as dd
from collections import Counter
from aux import find_free_filename, SHWriter


def count_labels(dataset_name):
    df = dd.io.load(os.path.join('./dataset_info', 'info_{}.h5'.format(dataset_name)))['df']
    return Counter(df.values.flatten())


def find_percentages(dataset_name):
    n_selected_per_label = (5, 10, 20, 30, 50, 100, 200)
    count = count_labels(dataset_name)
    min_val = min(count.values())

    frac_per_label = [npl * 1.0 / min_val for npl in n_selected_per_label]
    return frac_per_label


def get_configs():
    # general configurations
    # n_runs = 3
    n_runs = 1
    # heuristics = ('none', 'area', 'svm')
    heuristics = ('none', 'area', 'svm')# ,'clique_svm')
    verbose = True

    # # dataset configurations
    dataset_name = 'AwA2_vectors_train'
    dataset_dir = './dataset_info'
    impath = '/net/hci-storage02/groupfolders/compvis/datasets/Animals_with_Attributes2/single_folder_images'

    # # dataset configurations
    # dataset_name = 'Wikiart_Elgammal_EQ_artist_train'
    # dataset_dir = './dataset_info'
    # impath = '/export/home/pdamman/Wikiart_Elgammal'

    # dataset configurations
    # dataset_name = 'STL_label_train'
    # dataset_dir = './dataset_info'
    # impath = '/export/home/pdamman/STL10/img'

    # # dataset configurations
    # dataset_name = 'BreakHis_tumor_train'
    # dataset_dir = './dataset_info'
    # impath = '/export/home/kschwarz/Documents/Data/BreakHis/BreaKHis_v1/images'

    feature_dim = 512

    # heuristics configurations
    fac_corrected = 1
    n_selected_per_label = (5, 10, 20, 30, 50, 100, 200)
    n_min_clustering = 5      # need at least 4 to compute cluster score
    n_min_sampling = 3        # need at least 3 to compute cluster area for area approach
    # SVM heuristic configurations
    n_svm_iter = 3
    n_random_negatives_svm = 100
    weight_svm_random_negatives = 0.1
    n_wrong_threshold = 5

    weight_predictions = 0.7
    return locals()


outdir = './automated_runs/runs/generate_labels'
if not os.path.isdir(outdir):
    os.makedirs(outdir)


if __name__ == '__main__':
    config_vars = get_configs()
    config_vars['n_selected_per_label'] = find_percentages(config_vars['dataset_name'])
    prefix = 'cd ../../..\npython generate_labels.py'
    writer = SHWriter(prefix=prefix, verbose=True)
    writer.set_args(**config_vars)

    filename = os.path.join(outdir, '{}.sh'.format(config_vars['dataset_name']))
    filename = find_free_filename(filename)
    writer.write_sh(filename)

