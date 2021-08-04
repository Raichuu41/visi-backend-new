import md5 from 'md5';
import Router from 'express';
import db from '../config/db_secret.js';
import mysql from 'mysql2';

const router = Router();


const connection = mysql.createConnection({
    host: db.host,
    user: db.user,
    password: db.password,
    database: db.db,
});

connection.connect((err) => {
    if (!err) {
        console.log('Successfully connected to the database!');
    } else {
        console.log('Error connecting to the database');
        console.log(err);
    }
});


export { connection };
export default router;
