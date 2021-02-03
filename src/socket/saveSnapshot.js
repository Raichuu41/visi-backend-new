import fetch from 'node-fetch';
import { pythonApi } from '../config/pythonApi.js';

export default socket => async (data) => {
    console.log('Socket on: saveSnapshot');
    const {
        nodes, groups, dataset, count, userid, snapshotName, modelChanged, displayCount
    } = data;
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
                displayCount,
            }),
        })
            .then(response => response.text());
        socket.emit('BE-saveSnapshot', {status: 'success', snapshotName: snapshotName});
        console.log('Saved snapshot successfully!')
    } catch (err) {
        console.error('error - save snapshots python error');
        console.error(err);
        socket.emit('BE-saveSnapshot', {status: 'failed', error: err})
        console.log('Failed to save snapshot!')
    }
};
