import fetch from 'node-fetch';
import { promisify } from 'util';
import sharp from 'sharp';
// import graphMock from './mock/graphSmall'
// import exampleGraph from './mock/example_graph'
// import exampleNodes from './mock/exampleNodes';
import exampleNodes from '../mock/2582_sub_wikiarts';
// import { mergeLinksToNodes } from "./util/mergeLinksToNodes";
import { compareAndClean } from './util/compareAndClean';
import { getRandomColor } from './util/getRandomColor';
import trainSvm from './routes/trainSvm';
import stopSvm from './routes/stopSvm';
import pythonRoute from './routes/python/index';
import buildTripel from './util/buildTripels';

const express = require('express');
const fs = require('fs');

const kde2d = require('@stdlib/stdlib/lib/node_modules/@stdlib/stats/kde2d');

const mockDataLength = 50 //Object.keys(exampleNodes).length;


// const path = require('path');
const socket_io = require('socket.io');
// required for file serving
const app = express();

// const kdbush = require('kdbush');

const clusterfck = require('tayden-clusterfck');

const readFile = path =>
    new Promise((res, rej) => {
        fs.readFile(path, (err, data) => {
            if (err) {
                rej(err);
            } else {
                res(data);
            }
        });
    });


const colorTable = {
    0: '#e23511',
    1: '#1212d3',
    2: '#00cf09',
    3: '#9111ff',
    4: '#ff00c0',
    5: '#04fff0',
    6: '#ff8685',
    9: '#7cff6d',
    10: '#fffe6f',
    11: '#ff6af1',
    12: '#85feff',
};

const imgSizes = [
    10, // pixel
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
];

// Socket.io
const io = socket_io({ pingTimeout: 1200000, pingInterval: 300000 });
app.io = io;

const scaledPicsHash = {}; // scaled images in new archetecture 2

// const stringImgHash = {};       // normal (50,50) images in old architecture

const largeFileHash = {}; // the detailed images witch are loaded if needen

let nodesStore = {};

// let clusterStore = null;


// set different image path for prod/dev mode
let imgPath = '';

if (process.env.NODE_ENV === 'development') {
    imgPath = `${__dirname}/../images/2582_sub_wikiarts/`;
    // imgPath = `/export/home/kschwarz/Documents/Data/CUB_200_2011/images_nofolders/`;
} else {
//    imgPath = '/export/home/asanakoy/workspace/wikiart/images/';
    imgPath = '/export/home/kschwarz/Documents/Data/Wikiart_artist49_images/';
//    imgPath = `${__dirname}/images/`;
//    imgPath = `/export/home/kschwarz/Documents/Masters/Sabine_Project/images-de-piles/`;
//    imgPath = `/export/home/kschwarz/Documents/Data/tiny-imagenet-200/val/images/`;
}

if (process.env.NODE_ENV === 'development') {
    const timeFillImgDataCach = process.hrtime();
    // resizePics(imgPath, [110, 120, 130, 140, 150], exampleNodes)

    // fill scaledPicsHash

    console.time('fillImgDataCach');
    console.log(`fillImgDataCach of ${mockDataLength} files`);

    // generate dummy nodes
    for (let n = 0; n < mockDataLength; n += 1) {
        const i = n % mockDataLength;
        const node = exampleNodes[i];
        const pics = {};
        // TODO prüfen ob sich bilder auch als raw() abspeichern lassen
        Promise.all(imgSizes.map((size) => {
            const path = `${imgPath}${size}/${node.name}.png`;

            return sharp(path)
                .raw()
                .toBuffer({ resolveWithObject: true })
                .then(pic => pics[size] = pic);
        })).then(() => {
            if (!(n % 100)) {
                const diffFillImgDataCach = process.hrtime(timeFillImgDataCach);
                console.log(`${n}/${mockDataLength} pics cached took: ${diffFillImgDataCach[0] + diffFillImgDataCach[1] / 1e9}s`);
            }
            scaledPicsHash[node.name] = pics;
            if (n + 1 === mockDataLength) console.log('fillImgDataCach end');
        });

        //  }));

        // const path = `${imgPath}${node.name}.jpg`;
        // const pic = sharp(path);
        // Promise.all(imgSizes.map(async (size) => {
        //     // const file = await readFile(iconPath);
        //     pics[size] = await pic
        //         .resize(size, size)
        //         .max()
        //         .overlayWith(
        //             Buffer.alloc(4),
        //             { tile: true, raw: { width: 1, height: 1, channels: 4 } },
        //         )
        //         .raw()
        //         .toBuffer({ resolveWithObject: true });
        // })).then(() => {
        //     if (!(n % 100)) {
        //         const diffFillImgDataCach = process.hrtime(timeFillImgDataCach);
        //         console.log(`${n}/${mockDataLength} pics cached took: ${diffFillImgDataCach[0] + diffFillImgDataCach[1] / 1e9}s`);
        //     }
        //     scaledPicsHash[node.name] = pics;
        //     if (n + 1 === mockDataLength) console.log('fillImgDataCach end');
        // });
    }
}

/* app.use(bodyParser.json())
app.use(bodyParser.urlencoded({ extended: false })) */

app.use(express.json({ limit: '5mb' }));
app.use(express.urlencoded({ extended: false, limit: '5mb' }));
// app.use(cookieParser())


// console.log(process.env.NODE_ENV === 'development')

app.use('/', express.static('public'));

// app.use('/api/v1/users', users)
// TODO add python in route name and change frontend usage
app.post('/api/v1/trainSvm', trainSvm);
app.post('/api/v1/stopSvm', stopSvm);
app.use('/api/v1/', pythonRoute);
app.use('/api', express.static('images'));
/* app.get('/images/!*', (req, res) => {
    console.log(req.path)
    res.send()
}) */

//// set different image path for prod/dev mode
//let imgPath = '';
//
//if (process.env.NODE_ENV === 'development') {
//    // imgPath = `${__dirname}/images/images_3000/`;
//    imgPath = `/export/home/kschwarz/Documents/Data/CUB_200_2011/images_nofolders/`
//} else {
//    imgPath = '/export/home/asanakoy/workspace/wikiart/images/';
//}

if (!fs.existsSync(imgPath)) throw Error(`IMAGE PATH NOT EXISTS - ${imgPath}`);

io.sockets.on('connection', (socket) => {
    console.log('A user connected: ', socket.id);
    console.log('# sockets connected', io.engine.clientsCount);



    socket.on('requestImage', async (data) => {
        // console.log("requestImage")
        // console.log(data.name)
        const name = data.name;
        if (name) {
            try {
                let buffer;
                if (largeFileHash[name]) {
                    buffer = largeFileHash[name];
                } else {
                    const imagePath = `${imgPath}${name}.jpg`;
                    const file = await readFile(imagePath);
                    buffer = file.toString('base64');
                    largeFileHash[name] = buffer;
                }
                socket.emit('receiveImage', {
                    name,
                    buffer,
                    index: data.index,
                });
                console.log(`Image is send: ${name}`);
            } catch (err) {
                console.error(err);
            }
        } else {
            console.error('that shoud now happen - report please!!! (requests image withoutname');
        }
    });

    socket.on('updateNodes', async (data) => {
        console.log('updateNodes from client');
        // console.log(typeof data)
        // console.log(data)

        // first time data is empty (the client should send a empty object {})
        let updatedNodes = data.nodes; // || {}
        // /if(typeof updatedNodes !== 'object') updatedNodes = JSON.parse(updatedNodes)
        // updatedNodes = JSON.parse(updatedNodes)

        // the nodes object for mutating data before sending
        let nodes = {};

        // labels are scanned on serverside
        const labels = {};

        let categorys = [];

        // build tripel from data
        console.log('buildTripel');
        const tripel = buildTripel(updatedNodes);
        // console.log({ tripel });
        if (tripel) console.log(tripel);


        // before they should be cleaned and compared with maybe old data
        const time = process.hrtime();
        updatedNodes = compareAndClean(nodesStore, updatedNodes);
        const diff = process.hrtime(time);
        console.log(`CopareAndClean took ${diff[0] + diff[1] / 1e9} seconds`);


        if (process.env.NODE_ENV === 'development') {
            // const mockDataLength = 50 //Object.keys(exampleNodes).length;

            console.log(`nodes generated from mock #: ${mockDataLength}`);

            // generate dummy nodes
            for (let n = 0; n < mockDataLength; n += 1) {
                const i = n % mockDataLength;
                nodes[n] = exampleNodes[i];
            }

            // dummy categorys
            categorys = ['kat1', 'kat2', 'kat3'];
        } else {
            console.log('get nodes from python');

            try {
                const time2 = process.hrtime();
                const res = await fetch('http://localhost:8000/nodes', {
                    method: 'POST',
                    header: { 'Content-type': 'application/json' },
                    body: JSON.stringify({
                        nodes: updatedNodes,
                        tripel,
                    }),
                });
                // there are only nodes comming back from here
                const data = await res.json();
                console.log(data.categorys)
                nodes = data.nodes;
                categorys = data.categorys;
                const diff2 = process.hrtime(time2);
                console.log(`getNodesFromPython took ${diff2[0] + diff2[1] / 1e9} seconds`);
            } catch (err) {
                console.error('error - get nodes from python - error');
                console.error(err);
            }
        }

        // generate labels structure
        categorys.forEach((kat, i) => labels[i] = { name: kat, labels: [] });

        const nodeDataLength = Object.keys(nodes).length;
        console.log(nodeDataLength)
        socket.emit('totalNodesCount', nodeDataLength);


        // store data data for comparing later
        nodesStore = nodes;
        // console.log("this nodes are stored")
        // console.log(nodesStore)

        // if (process.env.NODE_ENV === 'development' && clusterStore) {
        //     nodes = clusterStore;
        // } else {
        // add default cluster value (max cluster/zooming)
        Object.values(nodes).forEach(node => node.cluster = nodeDataLength);

        // starting the clustering
        console.log('start clustering');
        const timeCluster = process.hrtime();
        const points = Object.values(nodes)
            .map((n, i) => {
                const point = [n.x, n.y]; // array with properties is ugly!
                point.id = i;
                point.x = n.x;
                point.y = n.y;
                return point;
            });

            // const kdtree = kdbush(points, n => n.x, n => n.y)
            // console.log("finish kdtree")

            // const smallBox = kdtree.range(-3, -3, 3, 3)//.map(id => nodes[id])
            // console.log(smallBox)
            // const middlebox = index.range(-10, -10, 10, 10).map(id => nodes[id])
        const hcCluster = clusterfck.hcluster(points);
        console.log('finish hccluster');

        const zoomStages = 20;
        const nodesPerStage = Math.round(nodeDataLength / zoomStages) || 1; // small #nodes can result to 0
        for (let i = 1; i <= nodeDataLength; i += nodesPerStage) {
            hcCluster.clusters(i).forEach((cluster, i) => {
                const agentId = cluster[0].id;
                // the user can change the amount of clusters
                if (nodes[agentId].cluster > i) nodes[agentId].cluster = i;
                // console.log(`${i}. first items has id: ${clust[0].id}`)
            });
            console.log(`Building ${i} clusters finished`);
        }
        console.log('finish clusters');

        const diffCluster = process.hrtime(timeCluster);
        console.log(`end clustering: ${diffCluster[0] + diffCluster[1] / 1e9} seconds`);
        // clusterStore = nodes;
        // }


        // calc kernel density estimation
        const timeKde = process.hrtime();

        const x = [];
        const y = [];
        Object.values(nodes).forEach((node) => {
            x.push(node.x);
            y.push(node.y);
        });

        const out = kde2d(x, y, {
            xMin: -20,
            xMax: 20,
            yMin: -20,
            yMax: 20,
            // 'h': [ 0.01, 255 ], // bandwith - schätze damit kann man die range der dichte angeben
            // 'n': 5 // default 25 - was ist das
        });
        // console.log(out)
        // console.log(out.z)


        const diffKde = process.hrtime(timeKde);
        console.log(`end clustering: ${diffKde[0] + diffKde[1] / 1e9} seconds`);


        // saving used colorKeys
        const colorKeyHash = {};

        // saving used colors for label
        const colorHash = {};

        const timeStartSendNodes = process.hrtime();

        // doing everything for each node and send it back
        Promise.all(Object.values(nodes)
            .map(async (node, i) => {
                // that this is not inside !!! DONT FORGET THIS
                // console.time('map' + i)
                // console.log("start")
                node.index = i;
                node.positives = [];
                node.negatives = [];

                // catch if there is no rank
                if (!node.rank && node.rank !== 0) node.rank = -1;

                if (!node.cluster) node.cluster = nodeDataLength;

                // setting color based on label
                if (colorHash[node.label]) {
                    node.color = colorHash[node.label];
                } else {
                    const index = Object.keys(colorHash).length;
                    colorHash[node.label] = colorTable[index];
                    node.color = colorHash[node.label];
                }

                // get a unique color for each node as a key
                while (true) {
                    const colorKey = getRandomColor();
                    if (!colorKeyHash[colorKey]) {
                        node.colorKey = colorKey;
                        colorKeyHash[colorKey] = node;
                        break;
                    }
                }

                // labels
                if (process.env.NODE_ENV === 'development') {
                    const n = categorys.length;
                    node.labels = [];
                    for (let i = 0; i < n; i++) node.labels.push(Math.random() >= 0.5 ? `${categorys[i]}_label_${i}` : null);
                }

                // check all labels for a list of all labels in UI
                node.labels.forEach((label, i) => label && (labels[i].labels.indexOf(label) === -1) && labels[i].labels.push(label));


//                const iconPath = `${imgPath}${node.name}.JPEG`;
                const iconPath = `${imgPath}${node.name}.jpg`;

                node.pics = {};
                node.cached = false; // this is interesting while performance messearuing
                node.url = `/images_3000/${node.name}.jpg`;

                try {
                    if (scaledPicsHash[node.name]) {
                        // node.buffer = iconsFileHash[node.name].buffer;
                        node.pics = scaledPicsHash[node.name];
                        // node.buffer = stringImgHash[node.name];
                        nodes.cached = true;
                    } else {
                         const file = await readFile(iconPath);
                        // console.log(file);
                        //
                        const buffer = await sharp(file)
                       //     .resize(50, 50)
                        //    .resize(200, 200)
                            .max()
                        //    .toFormat('jpg')
                        //    .toBuffer();
                        //node.buffer = `data:image/jpg;base64,${buffer.toString('base64')}`;// stringImgHash[node.name] = node.buffer; // save for faster reload TODO test with lots + large image
// new architecture 2

                        await Promise.all(imgSizes.map(async (size) => {                        node.pics[size] = await sharp(iconPath)
                                .resize(size, size)
                                .max()
                                .overlayWith(
                                    Buffer.alloc(4),
                                    { tile: true, raw: { width: 1, height: 1, channels: 4 } },
                                )
                                .raw()
                                .toBuffer({ resolveWithObject: true });
                        }));
                        scaledPicsHash[ node.name ] = node.pics;

                        // new archetecture 1
                        /* await Promise.all(arr.map(async (size) => {
                            const buffer = await sharp(file)
                                .resize(size, size)
                                .max()
                                .toFormat( 'jpg')
                        .toBuffer();node.pics[size] = `data:image/jpg;base64,${buffer.toString('base64')}`; // save for faster reload TODO test with lots + large image
                        }));*/
                    }

                    socket.compress(false).emit('node', node);
                    // console.timeEnd('map' + i)
                    // if(!node.pics) console.log("HJEQWERIHWQR")

                    if ((i + 1) % 100 === 0) {
                        const diffStartSendNodes = process.hrtime(timeStartSendNodes);
                        console.log(`node is send: ${node.name} #${node.index} after: ${diffStartSendNodes[0] + diffStartSendNodes[1] / 1e9}s`);
                        // socket.compress(false).emit('nodesCount', node.index);
                    }
                } catch (err) {
                    console.log('Node was not send cause of missing image - how to handle?');
                    console.error(err);
                    console.log(node.index);
                }
            })).then(() => {
            const diffStartSendNodes = process.hrtime(timeStartSendNodes);
            console.log(`all ${nodeDataLength} nodes send after: ${diffStartSendNodes[0] + diffStartSendNodes[1] / 1e9}s`);
            // console.log(a)
            socket.emit('allNodesSend');

            // socket.emit('updateKdtree', kdtree)

            // sending back the labels and the colors
            socket.emit('updateLabels', labels);
            console.log('color labels send');
        });
    });

    socket.on('disconnect', (reason) => {
        console.log('disconnect: ', socket.id);
        console.log('# sockets connected', io.engine.clientsCount);
        console.log(`reason: ${reason}`);
    });
    socket.on('reconnection', (data) => {
        console.log(`recconected: ${socket.id}`);
        console.log(data);
    });
});

module.exports = app;

