#!/bin/bash

input_dataset="./input/"
output_dataset="./output/"

echo "Input dataset path: $input_dataset"
echo "Output dataset path: $output_dataset"


if [ -e $input_dataset ]
then
  echo "$input_dataset exists"
else
  mkdir -p $input_dataset
fi

# # comment for bgl_2k
# zipfile=BGL.tar.gz?download=1
# if [ -e "$input_dataset/$zipfile" ]
# then
#   echo "$input_dataset/$zipfile exists."
# else
#   wget https://zenodo.org/record/3227177/files/${zipfile} -P $input_dataset
#   tar -xvzf "$input_dataset/$zipfile" -C $input_dataset
# fi

# uncomment for bgl_2k
wget https://raw.githubusercontent.com/logpai/loghub/master/BGL/BGL_2k.log -P $input_dataset

if [ -e $output_dataset ]
then
  echo "$output_dataset exists"
else
  mkdir -p $output_dataset
fi