#!/bin/bash

# REMEMBER TO RUN THE FOLLOWING BEFORE RUNNING THIS SCRIPT:
# chmod +x ./install_vsc.sh

# Define the specific file to be extracted
file_to_extract="code-stable-x64-1730354220.tar.gz"

# download the zipped installer file.
wget -O "$file_to_extract" "https://update.code.visualstudio.com/1.95.1/linux-x64/stable"

# Extract the specified tar.gz file
tar -xzvf "$file_to_extract"

# Define the extracted folder name
extracted_folder="VSCode-linux-x64"

# Create a 'data' directory inside the extracted folder
mkdir -p "$extracted_folder/data"

echo "Directory 'data' created inside $extracted_folder"