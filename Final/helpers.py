import deepdish as dd


def load_data(feature_file, info_file, split=0):
    if not os.path.isfile(feature_file):
        raise RuntimeError('Feature file not found.')
    if not os.path.isfile(info_file):
        raise RuntimeError('Info file not found.')

    df = dd.io.load(info_file)['df']

    data = dd.io.load(feature_file)
    try:
        names, features = data['image_id'], data['feature']
    except KeyError:
        try:
            names, features = data['image_names'], data['features']
        except KeyError:
            names, features = data['image_name'], data['features']

    is_shape_dataset = 'ShapeDataset' in feature_file
    is_office_dataset = 'OfficeDataset' in feature_file
    is_bam_dataset = 'BAMDataset' in feature_file
    if is_shape_dataset:
        outdir = 'ShapeDataset'
        category = ['shape', 'n_shapes', 'color_shape', 'color_background']

        df = df[df['split'] == split]
        df.index = range(len(df))
    elif is_office_dataset:
        outdir = 'OfficeDataset'
        category = ['genre', 'style']
    elif is_bam_dataset:
        outdir = 'BAMDataset'
        category = ['content', 'emotion', 'media']
    else:
        outdir = 'Wikiart'
        category = ['artist_name', 'genre', 'style', 'technique', 'century']

    if not (names == df['image_id']).all():
        raise RuntimeError('Image names in info file and feature file do not match.')

    outdict = {'image_names': names,
               'features': features,
               'labels': {c: df[c] for c in category}}

    return outdict, outdir