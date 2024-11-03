#!/bin/bash

file="${HOME}/.dataset/bgl/"
output="${HOME}/.dataset/bgl/output/"

# Check if the directory exists
if [ ! -d "$output" ]; then
  # Create the directory if it does not exist
  mkdir -p "$output"
  echo "Directory $output created."

if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

cd $file
zipfile=BGL.tar.gz?download=1
wget https://zenodo.org/record/3227177/files/${zipfile} -P $file
tar -xvzf $zipfile