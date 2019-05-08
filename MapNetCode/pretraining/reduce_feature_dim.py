import argparse
import deepdish as dd
from sklearn.decomposition import PCA
import h5py
import pickle
import os

parser = argparse.ArgumentParser(description='Reduce dimensionality of features in given files with the same PCA '
                                             '(trained on trainset) to  given feature dimension.')

parser.add_argument('--feature_file_train', default='../features/MobileNetV2_info_elgammal_subset_train_imagenet.hdf5',
                    type=str, help='Path to hdf5 file containing features.')
parser.add_argument('--feature_file_val', default=None,#'../features/MobileNetV2_info_elgammal_subset_val_imagenet.hdf5',
                    type=str, help='Path to hdf5 file containing features.')
parser.add_argument('--feature_file_test', default=None,#'../features/MobileNetV2_info_elgammal_subset_test_imagenet.hdf5',
                    type=str, help='Path to hdf5 file containing features.')
parser.add_argument('--feature_dim', default=512, type=int, help='Dimensionality of output features.')

args = parser.parse_args()


if __name__ == '__main__':
    pca = PCA(n_components=args.feature_dim)

    train_df = dd.io.load(args.feature_file_train)
    print('Reduce dimensionality from {} to {}.'.format(train_df['features'].shape[1], args.feature_dim))
    train_df['features'] = pca.fit_transform(train_df['features'])

    outfile = args.feature_file_train.replace('.hdf5', '_{}.hdf5'.format(args.feature_dim))
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('features', train_df['features'].shape, dtype=train_df['features'].dtype, data=train_df['features'])
        f.create_dataset('image_names', train_df['image_names'].shape, dtype=train_df['image_names'].dtype, data=train_df['image_names'])
    print('Saved features to {}'.format(outfile))

    for ft_file in [args.feature_file_val, args.feature_file_test]:
        if ft_file is None:
            continue
        data_df = dd.io.load(ft_file)
        data_df['features'] = pca.transform(data_df['features'])

        outfile = ft_file.replace('.hdf5', '_{}.hdf5'.format(args.feature_dim))
        with h5py.File(outfile, 'w') as f:
            f.create_dataset('features', data_df['features'].shape, dtype=data_df['features'].dtype,
                             data=data_df['features'])
            f.create_dataset('image_names', data_df['image_names'].shape, dtype=data_df['image_names'].dtype,
                             data=data_df['image_names'])
        print('Saved features to {}'.format(outfile))

    # save the PCA
    outfile = args.feature_file_train.replace('.hdf5', '_PCA_{}.pkl'.format(args.feature_dim))
    with open(os.path.abspath(outfile), 'w') as f:
        pickle.dump(pca, f)
    print('Saved PCA to {}'.format(outfile))

