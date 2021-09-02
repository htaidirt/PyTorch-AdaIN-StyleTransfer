#!/bin/sh

echo "-- Loading necessary data"

if [[ ! -d ./data/wikiart ]]
then
    echo "Downloading wikiart dataset from AWS S3..."
    aws s3 sync s3://axa-climate-data-science-storage-sandbox/datasets/wikiart_flat/ ./data/wikiart
else
    echo "[SKIP] Dataset wikiart already downloaded"
fi

if [[ ! -d ./data/coco_images ]]
then
    echo "Downloading COCO dataset from AWS S3..."
    aws s3 sync s3://axa-climate-data-science-storage-sandbox/datasets/test2015/ ./data/coco_images
else
    echo "[SKIP] Dataset COCO already downloaded"
fi

if [[ ! -f ./resources/vgg_normalised.pth ]]
then
    echo "Downloading VGG19 resource..."
    aws s3 cp s3://axa-climate-data-science-storage-sandbox/resources/vgg_normalised.pth ./resources/vgg_normalised.pth
else
    echo "[SKIP] VGG19 file already downloaded"
fi

echo "-- Setting up python environment"
pip3 install -r requirements.txt

echo "-- Start the training"
python3 train.py
