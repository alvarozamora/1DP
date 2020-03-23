#!/bin/bash
#SBATCH --job-name=sod
#SBATCH --output=sodo
#SBATCH --error=sode
#SBATCH --time=5:00:00
#SBATCH -p iric
#SBATCH -N 1
#SBATCH --ntasks-per-node=16

source ~/env.sh
cd /scratch/PI/kipac/pizza/CS315b/Sod2/

regent P.rg -t 1 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 2 -p 1 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 3 -p 2 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 4 -p 3 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 5 -p 4 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 6 -p 5 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 7 -p 6 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 8 -p 7 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 9 -p 8 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 10 -p 9 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 11 -p 10 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 12 -p 11 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 13 -p 12- o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 14 -p 13 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 15 -p 14 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 16 -p 15 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 17 -p 16 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 18 -p 17 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 19 -p 18 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
regent P.rg -t 20 -p 19 -o 1 -ll:cpu 15 -ll:csize 60000 -ll:util 0
