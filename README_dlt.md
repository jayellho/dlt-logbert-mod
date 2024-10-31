# Pre-requisites
1. Docker


# Set-up
1. Download datasets by running shell script from folder `scripts`:
e.g. for HDFS dataset
```
sh download_hdfs.sh
```
2. Build Docker container:
```
docker compose build
```

3. Run Docker container:
```
docker compose up
```


# TO-DOS:
1. Generalise Docker container for other datasets - right now it's just for BGL's data_process.py.
2. Create Docker container for running logbert.
3. 