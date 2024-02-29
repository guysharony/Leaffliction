# #!/bin/bash

# download zip and extract
output_file="leaves.zip"

wget "https://cdn.intra.42.fr/document/document/17547/leaves.zip" -O "$output_file"

unzip "$output_file" -d ./datasets/

rm "$output_file"

# sort and organize directories
mkdir -p ./datasets/images/Apple
mkdir -p ./datasets/images/Grape

find ./datasets/images -maxdepth 1 -type d -name 'Apple_*' -exec mv {} ./datasets/images/Apple/ \;
find ./datasets/images -maxdepth 1 -type d -name 'Grape_*' -exec mv {} ./datasets/images/Grape/ \;
