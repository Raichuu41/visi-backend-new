import json
from os import listdir
import os.path as path

if __name__ == '__main__':
    image_folder = 'test_data'
    images_folder = path.abspath(path.join(__file__, '..'))
    output = 'test_data'
    all_files = [f for f in listdir(path.join(images_folder, image_folder))
                 if f.endswith('.png')]
    with open(path.join(images_folder, f'dataset_json/{image_folder}.json'), 'w') as f:
        json.dump(all_files, f, indent=4)
