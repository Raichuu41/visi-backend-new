import Router from 'express';
import fetch from 'node-fetch';
import { pythonApi } from '../../config/pythonApi.js';

const router = Router();

router.get('/', async (req, res, next) => {
    console.log('GET: snapshots');
    const { userid, dataset } = req.query;
    console.log(dataset, userid);
    if (!dataset || !userid) return next(new Error('dataset ID or User ID is missing'));

    if (process.env.NODE_ENV === 'development') {
        res.json([{
            nodes: {},
            groups: [],
            count: 500,
            createdAt: 'Mon Oct 28 2019 12:14:15 GMT+0100 (Mitteleuropäische Normalzeit)',
        }]);
    } else {
        try {
            const data = await fetch(`${pythonApi}/getSnapshots?userid=${userid}&dataset=${dataset}`).then(response => response.json());
            res.json(data);
        } catch (err) {
            console.error('error - loading snapshots - python error');
            console.error(err);
            next(err);
        }
        // res.json({ snapshots: [] });
    }

    // res.json({ snapshots: [] });
});

router.get('/load', async (req, res, next) => {
    console.log('GET: snapshots load')
    const snapshotId = req.query.snapshot;
    try {
        const data = await fetch(`${pythonApi}/loadSnapshot?snapshot=${snapshotId}`)
            .then(response => response.json());
        res.json(data);
    } catch (err) {
        console.error('error - loading snapshot data');
        console.error(err);
        next(err);
    }
})


router.post('/', async (req, res, next) => {
    console.log('POST: snapshots');

    const {
        nodes, groups, dataset, count, userid,
    } = req.body;
    /*
    console.log({
        nodes,
        groups,
        dataset,
        count,
        userid,
    });
    */
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
                    nodes, groups, dataset, count, userid,
                }),
            }).then(response => response.text());
            res.json({
                message: 'Snapshot saved',
            });
        } catch (err) {
            console.error('error - save snapshots python error');
            console.error(err);
            next(err);
        }
    }
});

export default router;
