import fetch from 'node-fetch';
import { pythonApi } from '../config/pythonApi.js';

export default socket => async (data) => {
    console.log('Socket on: getGroupNeighbours');
    const {
        threshold, positives, groupId, userId, negatives,
    } = data;
    const body = {
        threshold,
        positives,
        groupId,
        userId,
    };
    // no negatives => initial function call
    if (negatives) {
        body.negatives = negatives;
    }
    try {
        const time = process.hrtime();
        const response = await fetch(`${pythonApi}/getGroupNeighbours`, {
            method: 'POST',
            header: { 'Content-type': 'application/json' },
            body: JSON.stringify(body),
        })
            .then(response => response.json());
        const { group, neighbours: allNeighbours } = response;
        // sort the keys by highest score, slice the best X amount
        const sorted_neighbours_sliced = Object.keys(allNeighbours)
            .sort(function (a, b) {
                return allNeighbours[a] - allNeighbours[b];
            })
            .slice(0, +threshold);
        // convert the best keys from array back to object
        const final_neighbours = {};
        sorted_neighbours_sliced.forEach(key => final_neighbours[key] = allNeighbours[key]);
        console.log(sorted_neighbours_sliced);
        // get max score  for checking the scores
        const max = Math.max.apply(null, Object.keys(allNeighbours)
            .map(function (x) {
                return allNeighbours[x]
            }));
        socket.emit('BE-getGroupNeighbours',
            {
                group,
                neighbours: final_neighbours,
                allData: response,
                status: 'success'
            });
        const diff = process.hrtime(time);
        console.log(`getGroupNeighbours from python took ${diff[0] + diff[1] / 1e9} seconds`);
    } catch (err) {
        console.error('error - getGroupNeighbours python error');
        console.error(err);
        socket.emit('BE-getGroupNeighbours',
            {status: 'failed', error: err})
    }
};
