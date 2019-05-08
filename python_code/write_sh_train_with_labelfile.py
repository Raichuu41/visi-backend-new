import os
from aux import find_free_filename, SHWriter


def get_configs():
    # general configurations
    #heuristics = ('none', 'area', 'svm')
    heuristics = ('clique_svm', 'svm')
    verbose = True
    use_test = False
    use_pretrained = True
    n_layers = 2

    # dataset configurations
    dataset_name = 'STL_label_train'
    # dataset_name = 'AwA2_vectors_train'
    # dataset_name = 'Wikiart_Elgammal_EQ_artist_train'
    dataset_dir = './dataset_info'
    if use_pretrained:
        outdir = './automated_runs/pretrained/{}_layers/smaller_LR'.format(n_layers)
    else:
        outdir = './automated_runs'

    feature_dim = 512
    projection_dim = 2

    # training configurations
    weight_unlabeled = 0.3
    lr = 1e-4
    batch_size = 2000
    batch_frac_labeled = 0.7
    max_epochs = 500

    return locals()


outdir = './automated_runs/runs/train_model'
if not os.path.isdir(outdir):
    os.makedirs(outdir)


if __name__ == '__main__':
    config_vars = get_configs()
    label_file_dir = os.path.join('./automated_runs/generated_labels', config_vars['dataset_name'], 'weighted_clustersampling/larger_clusters/clique_svm_10')
    label_files = [os.path.join(label_file_dir, f) for f in os.listdir(label_file_dir)
                   if f.endswith('.h5')]
    filename = os.path.join(outdir, '{}.sh'.format(config_vars['dataset_name']))
    filename = find_free_filename(filename)

    for i, label_file in enumerate(label_files):
        config_vars['label_file'] = label_file
        prefix = 'cd ../../..\npython train_with_labelfile.py' if i == 0 else 'python train_with_labelfile.py'
        writer = SHWriter(prefix=prefix, verbose=True)
        writer.set_args(**config_vars)
        writer.write_sh(filename, mode='a')

