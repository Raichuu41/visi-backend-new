import argparse
import os
import deepdish as dd
import pandas as pd
import sys

from sklearn.svm import SVC


if not os.getcwd().endswith('MapNetCode'):
    os.chdir('/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode')
sys.path.append('.')

from helpers import load_data


parser = argparse.ArgumentParser(description='Compute Classification accuracy by training an SVM for each task.')
parser.add_argument('--info_train', default='wikiart_datasets/info_artist_train.hdf5',
                    type=str, help='Path to hdf5 file containing dataframe of training set.')
parser.add_argument('--info_val', default='wikiart_datasets/info_artist_train.hdf5',
                    type=str, help='Path to hdf5 file containing dataframe of validation set.')
parser.add_argument('--feature_train', type=str, help='Path to hdf5 file containing features of training set.')
parser.add_argument('--feature_val', type=str, help='Path to hdf5 file containing features of validation set.')
parser.add_argument('--tasks', type=str, help='Tasks on which to evaluate.')

# sys.argv = []
args = parser.parse_args()
# args.info_train = 'pretraining/wikiart_datasets/info_elgammal_subset_train.hdf5'
# args.info_val = 'pretraining/wikiart_datasets/info_elgammal_subset_val.hdf5'
# args.feature_train = 'features/MobileNetV2_info_elgammal_subset_train.hdf5'
# args.feature_val = 'features/MobileNetV2_info_elgammal_subset_val.hdf5'
# args.tasks = 'artist_name,genre,style'

if __name__ == '__main__':
    outpath = './pretraining/evaluation'
    if not os.path.isdir(outpath):
        os.makedirs(outpath)

    exp_name = args.feature_train.split('/')[-1].replace('_train.hdf5', '').replace('_val.hdf5', '')
    exp_name = 'classification_acc_{}'.format(exp_name)
    eval_dict = {}

    data_dict_train, _ = load_data(args.feature_train, args.info_train)
    data_dict_val, _ = load_data(args.feature_val, args.info_val)

    for category in args.tasks.split(','):
        clf = SVC(kernel='rbf', C=10.0, gamma='auto', probability=False)

        label = data_dict_train['labels'][category].dropna()
        data = data_dict_train['features'][label.index]
        print('train SVM...')
        clf.fit(data, label)

        # evaluate SVM
        label = data_dict_val['labels'][category].dropna()
        data = data_dict_val['features'][label.index]
        print('predict...')
        prediction = clf.predict(data)

        # evaluate prediction
        acc = (prediction == label.values).sum() * 1.0 / len(label)
        eval_dict[category] = [acc]

    df = pd.DataFrame.from_dict(eval_dict, orient='columns')
    df.index = [args.feature_val.split('/')[-1]]
    df.to_csv(os.path.join(outpath, exp_name + '.csv'), index=False)
    print('Saved {}'.format(os.path.join(outpath, exp_name + '.csv')))


def merge_eval():
    file_dir = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/evaluation'
    files = [f for f in os.listdir(file_dir) if f.endswith('.csv')]
    filenames = [f.split('.csv')[0] for f in files]
    dfs = [pd.read_csv(os.path.join(file_dir, f), index_col=False) for f in files]
    out_df = None
    for df, idx in zip(dfs, filenames):
        df.index = [idx]
        if out_df is None:
            out_df = df
        else:
            out_df = out_df.append(df)

    idx = out_df.index.values
    idx.sort()
    out_df = out_df.loc[idx]
    vals = out_df.values
    out_df['average'] = vals.mean(axis=1)
    out_df.to_csv(os.path.join(file_dir, 'evaluation.xlsx'))