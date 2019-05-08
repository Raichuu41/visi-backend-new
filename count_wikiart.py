import deepdish as dd
from collections import Counter
count = {'artist_name': None, 'genre': None, 'style': None}
for split in ['train', 'val', 'test']:
    info_file = '/export/home/kschwarz/Documents/Masters/WebInterface/MapNetCode/pretraining/wikiart_datasets/info_elgammal_subset_{}.hdf5'.format(split)
    df = dd.io.load(info_file)['df']
    for task in ['artist_name', 'genre', 'style']:
        max_display = len(df[task].dropna())
        cnt = Counter(df[task].dropna().values)
        if count[task] is None:
            count[task] = cnt
        else:
            for k, v in cnt.items():
                count[task][k] += v
for task in ['artist_name', 'genre', 'style']:
    count[task] = sorted(count[task].items(), key=lambda kv: kv[1])
    print(count[task])