#!/bin/sh
#PBS -N treevaewald
#PBS -q gpuq1
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:ngpus=1:mem=86gb
##Request 1 node 8 core,4 GPU and 64gb ram
#PBS -V
# User Directives
# Specify the folder path
folder_path="/home/junyi/code/treevae/models/experiments/waldvarient"

# Use the find command to locate empty directories within the specified folder
empty_dirs=$(find "$folder_path" -type d -empty)

# Loop through each empty directory and remove it using the rmdir command
for dir in $empty_dirs; do
    rmdir "$dir"
done