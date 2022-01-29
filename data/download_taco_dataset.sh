#!/bin/bash
INPUT=./images/all_image_urls.csv
while read line
do
    # Get first column
    url=$(echo $line | cut -d, -f1)
    # download image
    wget $url -P ./images/images
done < $INPUT
