// stuff that changes through env

export const pythonApi = (process.env.NODE_ENV === 'development') ? 'localhost' : 'localhost';      //129.206.117.75

export const mockDataLength = 50;
