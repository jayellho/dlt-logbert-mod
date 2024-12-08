# LogBERT: Log Anomaly Detection via BERT
### [ARXIV](https://arxiv.org/abs/2103.04475) 

This repository provides the implementation of Logbert for log anomaly detection. 
The process includes downloading raw data online, parsing logs into structured data, 
creating log sequences and finally modeling. 

![alt](img/log_preprocess.png)

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.8
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
An virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r ./environment/requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

An alternative is to create a conda environment:
```
    conda create -f ./environment/environment.yml
    conda activate logbert
```
Reference: https://docs.conda.io/en/latest/miniconda.html

## Experiment
Logbert and other baseline models are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS), [BGL](https://github.com/logpai/loghub/tree/master/BGL), and [thunderbird]() datasets

### Run on HDFS Dataset
```shell script

cd HDFS

sh init.sh

# process data
python data_process.py

#run our Experiment 1 on our model
python train_mod.py
------------------------------------------------------
**if you have a trained model already
python val_only_mod.py --experiment 1
------------------------------------------------------
python deploy_mod.py --experiment 1 --threshold {0.-1}

#run logbert
python logbert.py vocab
python logbert.py train
python logbert.py predict

```

### Run on Thunderbird Dataset
```shell script
cd TBird

sh init.sh

# process data
python data_process.py

# run base logbert
python logbert.py vocab
python logbert.py train
python logbert.py predict

# run modified logbert
python train_mod.py
------------------------------------------------------
# if you have a trained model already (runs experiment 1)
python val_only_mod.py --experiment 1
------------------------------------------------------
python deploy_mod.py --experiment 1

```

### Folders created during execution
```shell script 
~/.dataset //Stores original datasets after downloading
project/output //Stores intermediate files and final results during execution
```
