import fs from 'fs';
import Router from 'express';
import dataSet from '../../config/datasets.js';
import ss from 'stream-stream';
import path from 'path';
import {fileURLToPath} from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const router = Router();

// GET - /api/v1/dataset/all
router.get('/all', async (req, res) => {
    console.log('/dataset/all');
    const datasets = dataSet.map(({
        id, name, description, size,
    }) => {
        // check if byte file exists
        // const byteFileName = path.join(__dirname, '/../../../images/bin', `${name}#${count}.bin`);
        const exists = true; //= fs.existsSync(byteFileName);
        // console.log({ byteFileName, exists });
        return {
            id, name, description, size, exists,
        };
    });
    return res.json(datasets);
});

// GET - /api/v1/dataset/images/:id/:count
router.get('/images/:id/:count', async (req, res, next) => {
    try {
        console.log('request dataset stream');
        let contentSize = 0;
        let { id, count } = req.params;
        const { name, size } = dataSet.find(e => e.id === id);
        if (count > size) count = size;
        console.log({
            id, name, count, size,
        });
        const files = [];
        // if (!imgPath || !count) return next(new Error('keine gültige id oder name'));
        let i = 0;
        while (i < count) {
            // i = (i + 500) < count ? i + 500 : +count;
            // const fileName = `${name}#${i}.bin`;
            const fileName = `${name}#${i}.bin`;
            const filePath = path.join(__dirname, '../../../images/bin/', fileName);
            const stat = fs.statSync(filePath);
            contentSize += stat.size;
            files.push(filePath);
            i++;
        }

        res.writeHead(200, {
            'Content-Type': 'application/octet-stream',
            'Content-Length': contentSize,
        });

        const stream = ss();
        files.forEach((file) => {
            stream.write(fs.createReadStream(file));
        });
        stream.end();
        stream.pipe(res, { end: false });
    } catch (e) {
        console.log('error in GET /images/:id/:count');
        console.log(e);
        // check for wrong file path
        if (e.code === 'ENOENT') {
            const file = path.basename(e.path);
            next(new Error(`Couldn't find image file: ${file}`));
        } else next(e);
    }
});

// GET - /api/v1/dataset/nodes/:id
router.get('/nodes/:id', async (req, res, next) => {
    console.log('request dataset nodes');
    const { name, count } = dataSet.find(e => e.id === req.params.id);
    // console.log({ name, count });
    // if (!imgPath || !count) return next(new Error('keine gültige id oder name'));
    const fileName = `${name}#${count}.json`;
    const filePath = path.join(__dirname, '/../../../images/', fileName);

    // const stat = fs.statSync(filePath);
    // console.log(stat);

    res.sendFile(filePath, (err) => {
        if (err) next(err);
        else console.log('Sent:', fileName);
    });
});


export default router;
