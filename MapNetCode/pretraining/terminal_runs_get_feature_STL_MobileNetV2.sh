python get_feature.py --info_file /export/home/kschwarz/Documents/Data/STL/info_STL_train.hdf5 --im_path None --stat_file /export/home/kschwarz/Documents/Data/STL/data/STL_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../features/STL --model mobilenet_v2 --not_narrow --stl_dataset

python get_feature.py --info_file /export/home/kschwarz/Documents/Data/STL/info_STL_test.hdf5 --im_path None --stat_file /export/home/kschwarz/Documents/Data/STL/data/STL_train_mean_std.pkl --batch_size 16 --device 0 --feature_dim 512 --output_dir ../features/STL --model mobilenet_v2 --not_narrow --stl_dataset



# reduce feature dim
python reduce_feature_dim.py --feature_file_train ../features/STL/STLDataset_MobileNetV2_info_STL_train.hdf5 --feature_file_test ../features/STL/STLDataset_MobileNetV2_info_STL_test.hdf5
