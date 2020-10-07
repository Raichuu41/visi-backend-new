/*
   {
        id: String, unique id
        name: String, Name that is shown to the user
        description: String, also shown to the user
        imgPath: String, absolute path to dicts
        mockDataFile: String, path to mock file
        count: Number, use only if want to reduce the dataset to the first n elements. Most time creating a new one is the bedder way
    }
 */
import path from 'path';
import {fileURLToPath} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const dataSet = [
    {
        id: '099',
        name: '2582_sub_wikiarts_debug',
        description: '...',
        imgPath: path.join(__dirname, '../../images/2582_sub_wikiarts_debug'),
        size: 4,
    },
    {
        id: '003',
        name: 'STL_debug',
        description: '----',
        imgPath: path.join(__dirname, '../../images/STL_debug'),
        size: 1300,
    },
    {
        id: '004',
        name: 'images_3000',
        description: 'Images 3000 Dataset',
        imgPath: path.join(__dirname, '../../images/images_3000'),
        size: 11728,
    }
];

export default dataSet;

