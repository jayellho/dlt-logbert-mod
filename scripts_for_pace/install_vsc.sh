#!/bin/bash

# REMEMBER TO RUN THE FOLLOWING BEFORE RUNNING THIS SCRIPT:
# chmod +x ./install_vsc.sh

# Update package index
sudo apt update -y

# Install dependencies
sudo apt install -y wget gpg

# Download Microsoft GPG key and add it to the system
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -o root -g root -m 644 packages.microsoft.gpg /usr/share/keyrings/
rm packages.microsoft.gpg

# Add VS Code repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list

# Update package index and install VS Code
sudo apt update -y
sudo apt install -y code

echo "Visual Studio Code installed successfully on Ubuntu."