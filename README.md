# Instructions

## Prerequisites

1. MySQL database is running with the following tables available
   1. snapshots
      1. user_id
      2. snapshot_id
      3. snapshot_name
      4. created_at
      5. dataset_id
      6. modified_model
      7. count
      8. groups_count
      9. display_count

    2. user_accounts
       1. user_id
       2. user_name
       3. password
   
2. Python 3.8+ Interpreter (64-bit) is installed with all requirements
   - Run `pip install -r requirements.txt`
   - Optional: Create a separate virtual environment (e.g. venv, conda)
3. NodeJS 16+ is installed and available
   - Run `npm install` to install all required packages
4. Linux OS
5. NVIDIA GPU (optional)
7. Images for at least one dataset

## Run commands

### Useful Database commands

1. sudo service mysql start
    - Starts the MySQL database (after installation)
2. sudo mysql -u root -p
    - Enters the MySQL terminal as root user on initial setup
3. select user from mysql.user
    - Shows all the available MySQL users
4. create user xyz@localhost identified by 'your password'
    - Creates a user accessible from specified host with a defined password
5. show databases
    - Shows all the available databases which the logged in MySQl user can see
6. use [database name]
    - Switches to use the specified database
7. grant all privileges on [database name].* to xyz@localhost
    - Provides all privileges to the tables of the database for the specified user

### Creating the Image Lists
1. Run `python /images/make_image_list.py`
    - The dataset folder name can be specified inside the file
    - The dataset images have to be located under `images/datasetName`
        - `(.../datasetName/image1.jpg, .../datasetName/image2.png, ...)`
    - Supported file endings are `.jpg` and `.png`
2. The files will be created under `/images/dataset_json`

### Creating Binary files
**Note that the Image List has to be generated beforehand!**
1. Run `npm run debug_bin`
    - Executes the script `debug_bin` specified in `package.json`
    - Starts the binBuilder with additional debug information
2. The files will be created under `/images/bin`
    - File name pattern: `datasetName#imageIndex.bin` 
      (e.g.: `test#1.bin, test#2.bin, ...`)
      
### Creating the Initialization JSON (Python Machine Learning)
**Note that the Image List has to be generated beforehand!**
1. Run `python /python_code/generate_dataset_json.py` with following parameters
    - `-n datasetName`
    - `-i datasetName`
    - `-a`
    - Example: `python generate_dataset_json.py -n xxl_data -i xxl_data -a`
2. The files will be created under `/images/init_json`    


### Starting the Python Backend Server
1. Ensure that the Database Credentials are correctly set in `server.py`
2. Ensure that the hostname is correctly set in `server.py`
3. Run `python server.py` to run the development mode
4. Run `python server.py -prod` or `python server.py --production` to run the production mode

### Starting the Node Backend Server
1. Run `npm run dev` to run the development mode
2. Run `npm run prod` tor un the production mode