python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name genre_style --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/10-15-10-46_NarrowNet512_VGG_genre_style_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name artist_genre --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/10-15-10-47_NarrowNet512_VGG_artist_genre_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name artist_style --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/10-15-10-47_NarrowNet512_VGG_artist_style_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name artist_genre_style --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/11-17-19-31_NarrowNet512_VGG_artist_genre_style_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name artist_genre_style_media_century --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/11-17-23-01_NarrowNet512_VGG_artist_genre_style_media_century_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name artist --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/11-18-06-47_NarrowNet512_VGG_artist_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name genre --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/11-18-08-48_NarrowNet512_VGG_genre_model_best_mean_acc.pth.tar

python get_feature.py --info_file wikiart_datasets/info_elgammal_subset_test.hdf5 --im_path /export/home/kschwarz/Documents/Data/Wikiart_Elgammal/resize_224 --stat_file wikiart_datasets/info_elgammal_subset_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../evaluation/pretrained_features --model vgg16_bn --exp_name style --weight_file /export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/models/eval_mtl/11-18-18-38_NarrowNet512_VGG_style_model_best_mean_acc.pth.tar
