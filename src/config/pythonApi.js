let baseUrl;
if (process.env.NODE_ENV === 'production') {
    baseUrl = '129.206.117.194';
} else {
    baseUrl = 'localhost'
}

export const pythonApi = `http://${baseUrl}:8023`;
