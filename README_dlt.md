# Pre-requisites
1. conda

# Set-up
1. Install conda (if not previously installed)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
```

# Activate virtual environment.
Purpose of virtual environment:
- Maintain package consistency.
- Prevent package dependency issues.
- Isolate environment from current system environment.

1. Change directory into where `environment.yml` is.
```
cd environment
```

2. Create the environment `logbert` as specified in `environment.yml`.
```
conda env create -f environment.yml
```

3. Activate the environment `logbert`.
```
conda activate logbert
```

4. Deactivate the environment.
```
conda deactivate
```
5. TLDR; just run everything below:
```
cd environment
conda env create -f environment.yml
conda activate logbert
```

# Running the code.
- Code has to be run separately for each dataset - to be refactored in the future.
- Below is an example for the `BGL` dataset.

1. Change directory into the `BGL` dataset, then copy and rename `.env_dataset`.
```
cd BGL && cp .env_dataset .env
```

2. Create input and output folders and download datasets by running `init.sh`:
```
sh init.sh
```

3. Preprocess the datasets.
* NOTE: Look out for print statements that indicate what to set `MAX_SEQ_LEN` and `MAX_TOKENS` to.
```
python data_process.py
```
4. If `MAX_SEQ_LEN` is larger than 510, truncate.
```
python truncate_tokens.py
```
5. Train the BERT model.
* NOTE 1: Truncate all sequences in `train`/`test_normal`/`test_abnormal` to a maximum of 510 tokens.
* NOTE 2: Edit `MAX_SEQ_LEN` and `MAX_TOKENS` in `.env`. `MAX_SEQ_LEN` should be at most 510; `MAX_TOKENS` should be the maximum value of any token in `train`/`test_normal`/`test_abnormal`.
```
python train_mod.py
```

6. Perform validation.
```
python val_only_mod.py
```

7. Compute metrics.
```
python deploy_mod.py
```

# Miscellaneous

## Set-up using PACE cluster - for GPU compute.
1. Spin up PACE cluster VM and go into it - read GaTech guides if unsure how.
2. Install Visual Studio Code within VM:
```
cd scripts_for_pace
chmod +x setup_for_pace.sh
./setup_for_pace.sh
```
3. Run Visual Studio Code by going into `scripts_for_pace/VSCode-linux-x64` and clicking the executable `code`.

## Set-up for Docker.
1. Set-up Docker container.
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