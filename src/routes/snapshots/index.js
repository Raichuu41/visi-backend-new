import Router from 'express';
import fetch from 'node-fetch';
import { pythonApi } from '../../config/pythonApi.js';

const router = Router();

router.get('/', async (req, res, next) => {
    console.log('GET: snapshots');
    const { userid, dataset } = req.query;
    console.log(dataset, userid);
    if (!dataset || !userid) return next(new Error('dataset ID or User ID is missing'));
    try {
        const data = await fetch(`${pythonApi}/getSnapshots?userid=${userid}&dataset=${dataset}`)
            .then(response => response.json());
        res.json(data);
    } catch (err) {
        console.error('error - loading snapshots - python error');
        console.error(err);
        next(err);
    }
    // res.json({ snapshots: [] });

    // res.json({ snapshots: [] });
});

router.get('/load', async (req, res, next) => {
    console.log('GET: snapshots load');
    const {userid, snapshot} = req.query;
    try {
        const data = await fetch(`${pythonApi}/loadSnapshot?snapshot=${snapshot}&userid=${userid}`)
            .then(response => response.json());
        res.json(data);
    } catch (err) {
        console.error('error - loading snapshot data');
        console.error(err);
        next(err);
    }
});

router.get('/resetTempModel', async (req, res, next) => {
    console.log('GET: reset temporary model');
    const {userid} = req.query;
    try {
        const data = await fetch(`${pythonApi}/resetTempModel?userid=${userid}`)
            .then(response => response.json());
        res.json(data);
    } catch (err) {
        console.error('error - reset temporary moodel');
        console.error(err);
        next(err);
    }
});


router.post('/', async (req, res, next) => {
    console.log('POST: snapshots');

    const {
        nodes, groups, dataset, count, userid, snapshotName, modelChanged
    } = req.body;
    if (process.env.NODE_ENV === 'development') {
        res.json({
            message: 'Snapshot not saved in dev mode',
        });
    } else {
        console.log('send snapshot to python');
        try {
            await fetch(`${pythonApi}/saveSnapshot`, {
                method: 'POST',
                header: { 'Content-type': 'application/json' },
                body: JSON.stringify({
                    nodes,
                    groups,
                    dataset,
                    count,
                    userid,
                    snapshotName,
                    modelChanged,
                }),
            })
                .then(response => response.text());
            res.json({
                message: 'Snapshot saved successfully',
            });
        } catch (err) {
            console.error('error - save snapshots python error');
            console.error(err);
            next(err);
        }
    }
});

export default router;
