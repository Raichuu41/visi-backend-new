import { connection } from '../routes/login.js';
import fetch from 'node-fetch';
import {pythonApi} from '../config/pythonApi.js';
import md5 from 'md5';

export default socket => async (data) => {
    console.log('Socket on: login');
    const { user, password } = data;
    // await fetch(`${pythonApi}/checkTemporaryModels`); // trigger deletion of old temp models
    if (!user) {
        socket.emit('BE-login', {status: 'failed', error: 'User is missing'});
        return null;
    }
    if (!password) {
        socket.emit('BE-login', {status: 'failed', error: 'Password is missing'});
        return null;
    }
    if (connection.state === 'disconnected') {
        socket.emit('BE-login', {status: 'failed', error: 'Database connection is missing'});
        return null;
    }
    try {
        connection.query('SELECT * FROM user_accounts WHERE user_name = ? ', [user], (error, results, fields) => {
            console.log(results);
            if (error) {
                socket.emit('BE-login', {status: 'failed', error: error});
                return null;
            }
            if (results.length > 0) {
                // check pw
                if (results[0].password !== md5(password)) {
                    socket.emit('BE-login', {status: 'failed', error: 'Incorrect password'})
                    return null;
                }
                socket.emit('BE-login', {status: 'success', isAuth: true, id: results[0]['user_id'],
                    user: results[0]['user_name']});
                return null;
            }
            socket.emit('BE-login', {status: 'failed', error: 'Username not found'});
            return null;
        });
    } catch (e) {
        socket.emit('BE-login', {status: 'failed', error: e});
        return null;
    }
};
