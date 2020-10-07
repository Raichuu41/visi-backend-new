import fs from 'graceful-fs';
import path from 'path';
import imgSizes from './imgSizes.js';
import sharp from 'sharp';
import cliProgress from 'cli-progress';
import dataSet from './datasets.js';

const loadJson = () => {
    const dirname = path.resolve();
    dataSet.forEach(function (dataset_element) {
        const dataSetName = dataset_element['name']
        fs.readFile(path.join(dirname, `../../images/dataset_json/${dataSetName}.json`), async (error, data) => {
            if (error) {
                console.log('Async Read : NOT successful');
                console.log(error);
            } else {
                try {
                    const dataJson = JSON.parse(data);
                    const progressBar = new cliProgress.SingleBar({}, cliProgress.Presets.shades_classic);
                    progressBar.start(Object.keys(dataJson).length, 0);
                    await Promise.all(dataJson.map(async (sourceFile, idx) => {
                        let writeStream;
                        // if (writeStream) writeStream.end();
                        const binFilePath = path.join(dirname,
                            `../../images/bin/${dataSetName}#${idx}.bin`);
                        writeStream = fs.createWriteStream(binFilePath);
                        const imageFilePath = fs.realpathSync(path.join(dirname,
                            `../../images/${dataSetName}/${sourceFile}`));
                        const pics = Object.create(null);
                        await Promise.all(imgSizes.map(async (size) => {
                            pics[size] = await sharp(imageFilePath)
                                .resize(size, size, { fit: 'inside' })
                                .ensureAlpha()
                                .raw()
                                .toBuffer({ resolveWithObject: true });
                        }));
                        // console.log("Values: ", Object.values(pics));
                        Object.values(pics)
                            .forEach((p) => {
                                if (p) {
                                    writeStream.write(Buffer.from([p.info.width, p.info.height]));
                                    writeStream.write(p.data);
                                }
                            });
                        writeStream.end();
                        progressBar.increment(1);
                    }));
                    progressBar.stop();
                } catch (error) {
                    console.log(error);
                }
            }
        });
    });
};

loadJson();
