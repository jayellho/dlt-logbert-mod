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
**NOTE: This is due to BERT's max token limit being 512, and less 2 to cater for the 2 special tokens ([CLS] and [SEP]).**

5. Train the BERT model.
* NOTE 1: Truncate all sequences in `train`/`test_normal`/`test_abnormal` to a maximum of 510 tokens.
* NOTE 2: Edit `MAX_SEQ_LEN` and `MAX_TOKENS` in `.env`. `MAX_SEQ_LEN` should be at most 510; `MAX_TOKENS` should be the maximum value of any token in `train`/`test_normal`/`test_abnormal`. Run `python get_max_token_and_others.py` to print the `MAX_TOKEN` value.
```
python train_mod.py
```

6. Perform validation.
```
python val_only_mod.py
```

7. Compute metrics.
* NOTE: set `THRESHOLD` in `.env` to output from running `val_only_mod.py`.
```
python deploy_mod.py
```

# Experiments Conducted
1. Geometric mean over token probabilities
2. Geometric mean over token anomaly scores (i.e. 1 - probabilities)
3. Harmonic mean between approaches (1) and (2)
4. Approach (3) over only the top 40% most anomalous tokens

# Miscellaneous

## Description of helper scripts
Found in `misc/helper_scripts` folder.
1. `filter_oov....py`: 
- Gets population of tokens from `train` file
- Removes lines from `test_abnormal` and `test_normal` files that have >= 1 token not in the abovementioned population.
- Writes to `test_abnormal_filtered` and `test_normal_filtered`.
- NOTE: Deals with two inputs - one for files formatted as lines of integers (`filter_oov_by_int.py`) and another for those formatted as lines of hexadecimals (`filter_oov_by_hexa.py`).

2. `truncate_tokens.py`:
- Reads in lines from a file - assumes items within line are **space**-separated.
- Truncates lines beyond user-specified limit - this is to deal with BERT max token limitations (512, including special [CLS] and [SEP] tokens).
- Writes to an output file (does not affect original input file).

3. `get_best_threshold.py`:
- Not very useful - to refactor to automatically explore multiple thresholds.
- Currently, just computes F1, Precision and Recall for a user-specified threshold.

4. `get_max_token_and_others.py`:
- Gets the max token value for updating `MAX_TOKENS` value in `.env`. This value represents the vocabulary size for a given `train`/`test_normal`/`test_abnormal` file.
- Gets other measures: number of lines more than a given sequence limit, average length of sequence by number of tokens.

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