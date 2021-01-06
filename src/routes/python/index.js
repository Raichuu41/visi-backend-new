import Router from 'express';
import fetch from 'node-fetch';
import { compareAndClean } from '../../util/compareAndClean.js';
import buildLabels from '../../util/buildLabels.js';
import { getRandomUnusedId } from '../../util/getRandomUnusedId.js';
import { pythonApi } from '../../config/pythonApi.js';

const router = Router();

router.post('/updateLabels', async (req, res, next) => {
    console.log('updateLabels');
    const nodes = compareAndClean({}, req.body.nodes);
    console.log(nodes);

    if (process.env.NODE_ENV === 'development') {
        res.status = 200;
        res.send();
    } else {
        console.log('send updateLabels to python');
        try {
            const time = process.hrtime();
            await fetch(`${pythonApi}/updateLabels`, {
                method: 'POST',
                header: { 'Content-type': 'application/json' },
                body: JSON.stringify({ nodes, userId: req.body.userId }),
            }).then(response => response.json());
            const diff = process.hrtime(time);
            res.send();
            console.log(`updateLabels from python took ${diff[0] + diff[1] / 1e9} seconds`);
        } catch (err) {
            console.error('error - updateLabels python error');
            console.error(err);
            next(err);
        }
    }
});

// This is right now just for the python backend to get data back to UI without request
router.post('/updateEmbedding', async (req, res, next) => {
    console.log('POST /updateEmbedding');
    const { categories, nodes, socket_id } = req.body;

    if (!socket_id) return next(new Error('No socket connection'));

    // todo @Katja: why is categories not always inside?
    const labels = categories ? buildLabels(categories, nodes) : undefined;
    const socket = req.app.io.sockets.sockets[socket_id];
    if (!socket) return next(new Error(`No socket with ID: ${socket_id} found`)); // TODO maybe deliver error to frontend
    if (labels) socket.emit('updateCategories', { labels });
    // confirm is {stopped: true/false}for signaling if the user hast stopped
    socket.emit('updateEmbedding', { nodes }, (confirm) => {
        console.log(confirm);
        res.json(confirm);
    });
});

router.post('/startUpdateEmbedding', async (req, res, next) => {
    console.log('POST /startUpdateEmbedding');
    const { body } = req;
    // console.log({ body });
    const { socketId, nodes } = body;
    console.log({ socketId });
    // console.log(app)
    if (!socketId) return next(new Error('No Client ID delivered'));
    res.send();

    try {
        const time = process.hrtime();
        await fetch(`${pythonApi}/startUpdateEmbedding`, {
            method: 'POST',
            header: { 'Content-type': 'application/json' },
            body: JSON.stringify(body),
        }).then(response => response.text());
        const diff = process.hrtime(time);
        // console.log(data);
        // res.send(data);
        console.log(`startUpdateEmbedding from python took ${diff[0] + diff[1] / 1e9} seconds`);
    } catch (err) {
        console.error('error - startUpdateEmbedding python error');
        console.error(err);
        next(err);
    }
});

// todo ist this necessary if the sopped state is already transmitted?
router.post('/stopUpdateEmbedding', async (req, res, next) => {
    console.log('POST /stopUpdateEmbedding');
    const { body } = req;
    console.log({ body });
    const { socketId } = body;
    console.log({ socketId });
    // console.log(app)
    if (!socketId) return next(new Error('No Socket ID delivered'));

    try {
        const time = process.hrtime();
        const data = await fetch(`${pythonApi}/stopUpdateEmbedding`, {
            method: 'POST',
            header: { 'Content-type': 'application/json' },
            body: JSON.stringify(body),
        }).then(response => response.text());
        const diff = process.hrtime(time);
        res.send(data);
        console.log(`stopUpdateEmbedding from python took ${diff[0] + diff[1] / 1e9} seconds`);
    } catch (err) {
        console.error('error - stopUpdateEmbedding python error');
        console.error(err);
        next(err);
    }
});

export default router;
