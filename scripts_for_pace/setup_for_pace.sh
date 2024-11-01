#!/bin/bash

# This script does the following:
# 1) Install the portable version of Visual Studio Code.
# 2) Set-up existing Docker installation such that you can run without sudo. << DOESNT WORK YET.
# REMEMBER TO RUN THE FOLLOWING BEFORE RUNNING THIS SCRIPT:
# chmod +x ./setup_for_pace.sh

# Install portable version of VSC.
## define the specific file to be extracted
file_to_extract="code-stable-x64-1730354220.tar.gz"

## download the zipped installer file.
wget -O "$file_to_extract" "https://update.code.visualstudio.com/1.95.1/linux-x64/stable"

## extract the specified tar.gz file
tar -xzvf "$file_to_extract"

## define the extracted folder name
extracted_folder="VSCode-linux-x64"

## create a 'data' directory inside the extracted folder
mkdir -p "$extracted_folder/data"

echo "Directory 'data' created inside $extracted_folder"

# # Allow running Docker without sudo.
# ## create the Docker group if it doesn't already exist.
# sudo groupadd docker 2>/dev/null

# ## add the current user to the Docker group.
# sudo usermod -aG docker "$(whoami)"
# echo "User $(whoami) added to the Docker group."

# # Prompt user to log out and back in for changes to take effect.
# echo "Please log out and log back in, or run 'newgrp docker' to refresh your group membership."