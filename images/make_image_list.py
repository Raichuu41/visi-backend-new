import json
from os import listdir
import os.path as path

if __name__ == '__main__':
    image_folder = 'test_30'
    images_folder = path.abspath(path.join(__file__, '..'))
    output = 'test_30'
    all_files = [f for f in listdir(path.join(images_folder, image_folder))
                 if f.endswith('.png') or f.endswith('.jpg')]
    with open(path.join(images_folder, f'dataset_json/{image_folder}.json'), 'w') as f:
        json.dump(sorted(all_files), f, indent=4)
