#!/bin/bash

# Generate a file list for Makefile to include
# ${1} is the directory
# Take Runs 900 -- 2000

rl=$(ls -1 ${1} | sed -nr 's/run0000((09|1[[:digit:]])[[:digit:]]{2})/\1/p')
echo rl:=${rl}
for r in ${rl}; do
    echo srl-${r}:=$(ls -1 ${1}/run0000${r}/PreAnalysis_*_File*.root | cut -d_ -f 4 | sed -r 's/File([0-9]*).root/\1/')
done
    
