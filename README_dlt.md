# Pre-requisites
1. Docker

# Set-up using PACE cluster - for GPU compute.
1. Spin up PACE cluster VM and go into it - read GaTech guides if unsure how.
2. Install Visual Studio Code within VM:
```
cd scripts_for_pace
chmod +x setup_for_pace.sh
./setup_for_pace.sh
```
3. Run Visual Studio Code by going into `scripts_for_pace/VSCode-linux-x64` and clicking the executable `code`.


# Set-up
1. Download datasets by running shell script from folder `scripts`:
e.g. for HDFS dataset
```
sh download_hdfs.sh
```
2. Set-up Docker container.
- Change directory to folder `docker`: 
```
cd docker
```
- Rename `.env_docker` to `.env`.
- Build Docker container:
```
docker compose build
```
- Run Docker container:
```
docker compose up
```


# TO-DOS:
1. Generalise Docker container for other datasets - right now it's just for BGL's data_process.py.
2. Create Docker container for running logbert.
3. 