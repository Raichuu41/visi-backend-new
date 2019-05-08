import torch
import os
import pandas as pd


def get_classes(filename):
    modelname = 'VGG_' if 'VGG' in filename else 'MobileNetV2_'
    classes = filename.split('/')[-1].split(modelname)[-1].split('_model_best')[0]
    return classes.split('_')

modeldir = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl'
modelname = 'VGG'
infix = 'NarrowNet512_{}'.format(modelname)
files = [os.path.join(modeldir, f) for f in os.listdir(modeldir) if (
    infix in f and
    f.endswith('model_best.pth.tar')
)]
mean_files = [os.path.join(modeldir, f) for f in os.listdir(modeldir) if (
    infix in f and
    f.endswith('mean_acc.pth.tar')
)]
files.sort()
mean_files.sort()


outfile = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/evaluation_multitask_{}.xlsx'.format(modelname)

names = [', '.join(get_classes(f)) for f in files]
mean_names = [', '.join(get_classes(f)) + ' (mean_acc)' for f in mean_files]
#
# outdf = pd.DataFrame(columns=['artist_name', 'genre', 'style', 'mean'], index=names)
#
#
# for f, name in zip(files+mean_files, names + mean_names):
#     try:
#         best = torch.load(f, map_location='cpu')
#     except IOError:
#         print('Did not find {}'.format(f))
#         continue
#     acc_dict = best['class_acc']
#     acc_dict['mean'] = best['acc']
#     for k, v in acc_dict.items():
#         outdf.at[name, k] = v
#
# outdf.to_excel(outfile)



# write .sh file
sh_file = '../pretraining/terminal_runs_get_feature_MTL_{}.sh'.format(modelname)

with open(sh_file, 'w') as f:
    for file in mean_files:
        exp_name = '_'.join(get_classes(file))
        modelselection = 'mobilenet_v2' if modelname == 'MobileNetV2' else 'vgg16_bn'
        line = 'python get_feature.py ' \
                '--info_file wikiart_datasets/info_elgammal_subset_val.hdf5 ' \
                '--im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 ' \
                '--stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl ' \
                '--batch_size 16 --device 0 --feature_dim 512 ' \
                '--output_dir ../evaluation/pretrained_features ' \
                '--model {} ' \
                '--exp_name {} ' \
                '--weight_file {}'.format(modelselection, exp_name, os.path.join(modeldir, file))

        f.write(line + '\n\n')