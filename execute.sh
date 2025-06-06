#!/bin/bash

echo "compiling file : $1 to kernel.sh"
echo "===================="
nvcc $1 -o kernel.sh
./kernel.sh
echo
echo "===================="
echo "deleting kernel.sh"
rm kernel.sh 