import fetch from 'node-fetch';
import { getRandomColor } from '../util/getRandomColor.js';
import { buildLabels } from '../util/buildLabels.js';
import { pythonApi } from '../config/pythonApi.js';
import dataSets from '../config/datasets.js';
import { devMode } from '../config/env.js';

// on: getNodes
export default socket => async (data) => {
    console.log('getNodes');
    const nodes = {}; // the nodes object for mutating differently in dev mode
    let categories = [];
    let time = 0;
    let rangeX;
    let rangeY;
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    const {
        datasetId, userId, count, init, nodesFromSnapshot,
    } = data;
    console.log({
        datasetId,
        userId,
        count,
        init,
        nodesFromSnapshot,
    });
    const dataset = dataSets.find(e => e.id === datasetId);
    if (!dataset) {
        // TODO Error handling, maybe a error emit
        console.error('No valid dataset');
        console.error({
            dataset,
            datasetId,
            dataSets
        });
        return socket.emit('Error', { message: 'Invalid datasetId' });
    }

    try {
        console.log('get init nodes from python');
        const time2 = process.hrtime();

        // check counts of all datasets
        // await Promise.all(dataSets.map(async (d) => {
        //     const all = await fetch(`${pythonApi}/getNodes?dataset=${d.name}`).then(res => res.json());
        //     const length = Object.keys(all.nodes).length;
        //     console.log(d.name, length);
        // }));
        const jsonAll = await fetch(`${pythonApi}/getNodes?dataset=${dataset.name}`)
            .then(res => res.json());
        /*
        const dirname = path.resolve();
        console.log(path.join(dirname, `/images/${dataset.name}.json`));
        const jsonAll = fs.readFileSync(path.join(dirname, `/images/${dataset.name}.json`), 'utf8', (error, data) =>{
            return JSON.parse(data);
        })
         */
        const jsonNodes = jsonAll.nodes;
        const keys = Object.keys(jsonNodes);
        console.log(`get #${keys.length} nodes from init`);

        if (!nodesFromSnapshot) {
            keys.forEach((key) => {
                // maybe count is higher but than max nodes in dataset will automatically the highest
                if (jsonNodes[key].idx < count) {
                    nodes[jsonNodes[key].idx] = {
                        x: jsonNodes[key].x,
                        y: jsonNodes[key].y,
                        name: key,
                        label: jsonNodes[key].label,
                        labels: [],
                        index: jsonNodes[key].idx,
                    };
                    if (jsonNodes[key].x > maxX) maxX = jsonNodes[key].x;
                    if (jsonNodes[key].x < minX) minX = jsonNodes[key].x;
                    if (jsonNodes[key].y > maxY) maxY = jsonNodes[key].y;
                    if (jsonNodes[key].y < minY) minY = jsonNodes[key].y;
                }
            });
        } else {
            Object.keys(nodesFromSnapshot)
                .forEach((i) => {
                    nodes[i] = nodesFromSnapshot[i];
                    if (nodes[i].x > maxX) maxX = nodes[i].x;
                    if (nodes[i].x < minX) minX = nodes[i].x;
                    if (nodes[i].y > maxY) maxY = nodes[i].y;
                    if (nodes[i].y < minY) minY = nodes[i].y;
                });
        }
        rangeX = Math.abs(maxX - minX);
        rangeY = Math.abs(maxY - minY);
        console.log({
            minX,
            maxX,
            minY,
            maxY,
            rangeX,
            rangeY,
        });
        Object.keys(nodes)
            .forEach((k) => {
                nodes[k].x = (((nodes[k].x - minX) / rangeX) * 30) - 15;
                nodes[k].y = (((nodes[k].y - minY) / rangeY) * 30) - 15;
            });
        const diff2 = process.hrtime(time2);
        time = diff2[0] + (diff2[1] / 1e9);
        console.log(`get init nodes from json took ${time} seconds`);
    } catch (e) {
        console.error('Error while getting nodes from python/json files');
        console.error(e.message);
        console.error(e);
        socket.emit('error', 'Failed to load init data from json file.');
    }

    const nodesLength = Object.keys(nodes).length;
    socket.emit('totalNodesCount', { count: nodesLength });

    if (devMode) {
        // dummy category's
        categories = ['kat1', 'kat2', 'kat3'];
        const c = categories.length;

        // generate dummy nodes
        for (let n = 0; n < nodesLength; n += 1) {
            // generate dummy labels
            nodes[n].labels = [];
            for (let i = 0; i < c; i += 1) {
                nodes[n].labels.push(Math.random() >= 0.5 ? `${categories[i]}_label_${i}` : null);
            }
        }
        // build and sending back labels (- )labels are scanned on server-side)
        socket.emit('updateCategories', { labels: buildLabels(categories, nodes) });
        console.log('updateCategories: labels are send');
        socket.emit('initPython', { done: true });
    } else {
        try {
            const time1 = process.hrtime();
            fetch(`${pythonApi}/nodes`, {
                method: 'POST',
                header: { 'Content-type': 'application/json' },
                body: JSON.stringify({
                    dataset: dataset.name,
                    count,
                    userId,
                    init,
                    nodes,
                    // tripel,
                }),
            })
                .then(async (res) => {
                    if (res.ok) {
                        try {
                            const data2 = await res.json();

                            if (data2.categories) categories = data2.categories;
                            socket.emit('updateCategories', { labels: buildLabels(categories, nodes) });
                            const diff1 = process.hrtime(time1);
                            time = diff1[0] + (diff1[1] / 1e9);
                            if (data2.nodes) {
                                socket.emit('updateEmbedding', {
                                    nodes: data2.nodes,
                                    time
                                });
                            }
                            socket.emit('initPython', data2);
                        } catch (err) {
                            // JSON Error here?
                            console.error('fetch works but response is not working - why?');
                            console.log(err);
                            console.log(res);
                            // socket.emit('sendAllNodes', nodes);
                            socket.emit('Error', {
                                message: err.message,
                                err,
                                res
                            });
                        }
                    }
                });
            // there are only nodes comming back from here
        } catch (err) {
            // todo bedder error handling, return and emit to inform user
            console.error('error - get nodes from python - error');
            console.error(err);
            socket.emit('Error', {
                message: 'error - get nodes from python - error',
                err
            });
            // todo remove after right loading from file
            // const diffStartSendNodes = process.hrtime(timeStartSendNodes);
            // console.log(`all ${nodeDataLength} nodes send after: ${diffStartSendNodes[0] + (diffStartSendNodes[1] / 1e9)}s`);
            // return socket.emit('sendAllNodes', nodes);
        }
    }

    // simulate missing nodes
    /* if (Object.keys(nodes).length < dataset.count) {
        let n = Object.keys(nodes).length; // maybe n = 1000, count = 2000
        while (n < dataset.count) {
            nodes[n - 1] = {
                index: n - 1,
                x: (Math.random() * 40) - 20,
                y: (Math.random() * 40) - 20,
                name: `mock node: ${n - 1}`,

            };
            n += 1;
        }
    } */

    // saving used colorKeys
    const colorKeyHash = {};
    // const timeStartSendNodes = process.hrtime();

    // doing everything for each node and send it back
    Object.values(nodes)
        .map(async (node) => {
            // get a unique color for each node as a key
            while (true) {
                const colorKey = getRandomColor();
                if (!colorKeyHash[colorKey]) {
                    node.colorKey = colorKey;
                    colorKeyHash[colorKey] = node;
                    break;
                }
            }

            // TODO in Python?
            if (!node.clique) node.clique = [1, 2, 3];
            if (!node.rank && node.rank !== 0) node.rank = 0.5;
        });

    console.log('sendAllNodes');
    return socket.emit('sendAllNodes', {
        nodes,
        time
    });
};
