#!/usr/bin/env bash

# download data and convert to .json format

if [ ! -d "data/raw/img_align_celeba" ] || [ ! "$(ls -A data/raw/img_align_celeba)" ] || [ ! -f "data/raw/list_attr_celeba.txt" ]; then
	echo "Please download the celebrity faces dataset and attributes file from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
	exit 1
fi

if [ ! -f "data/raw/identity_CelebA.txt" ]; then
	echo "Please request the celebrity identities file from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html"
	exit 1
fi

if [ ! -d "data/all_data_new" ] || [ ! "$(ls -A data/all_data_new)" ]; then
	echo "Preprocessing raw data"
	python preprocess/metadata_to_json_new.py
fi

NAME="celeba" # name of the dataset, equivalent to directory name

cd ../utils

./preprocess_new.sh --name $NAME $@

cd ../$NAME
