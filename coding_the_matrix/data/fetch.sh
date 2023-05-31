#!/bin/bash

download_resource() {
    local url="$1"
    local output_file="$2"
    curl -s -o "$output_file" "$url"
}

urls=(
    http://resources.codingthematrix.com/stories_small.txt
    http://resources.codingthematrix.com/stories_big.txt
    http://resources.codingthematrix.com/img01.png
    http://resources.codingthematrix.com/voting_record_dump109.txt
    http://resources.codingthematrix.com/US_Senate_voting_data_109.txt
    http://resources.codingthematrix.com/UN_voting_data.txt
    http://resources.codingthematrix.com/UN_voting_data.txt
    http://resources.codingthematrix.com/board.png
    http://resources.codingthematrix.com/cit.png
    http://resources.codingthematrix.com/train.data
    http://resources.codingthematrix.com/validate.data
    http://resources.codingthematrix.com/age-height.txt
    http://resources.codingthematrix.com/flag.png
    http://resources.codingthematrix.com/Dali.png
    http://resources.codingthematrix.com/game_results.csv
    http://resources.codingthematrix.com/mnist-images.dat
    http://resources.codingthematrix.com/mnist-labels.dat
    http://resources.codingthematrix.com/faces.zip
    http://resources.codingthematrix.com/unclassified.zip
    http://resources.codingthematrix.com/indexindex.txt
    http://resources.codingthematrix.com/titles.txt
    http://resources.codingthematrix.com/links.bin
    http://resources.codingthematrix.com/inverseindex
    http://resources.codingthematrix.com/train.data
    http://resources.codingthematrix.com/validate.data
)

# Iterate over the URLs and download each resource
for url in "${urls[@]}"; do
    filename=$(basename "$url")
    echo "Downloading $filename..."
    download_resource "$url" "$filename"
    echo "Download of $filename completed."
done
