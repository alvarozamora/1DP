#!/bin/bash
#SBATCH --job-name=3p
#SBATCH --output=3po
#SBATCH --error=3pe
#SBATCH --time=4:00:00
#SBATCH -p iric
#SBATCH --nodes=1
#SBATCH -c 8

cd /scratch/PI/kipac/pizza/CS315b/Particle/


regent 3P.rg -ll:cpu 2 -ll:util 1 -ll:csize 30000 -lg:sched 4 -lg:window 5 #-ll:dma 1 -lg:prof 1 -lg:prof_logfile strong1.gz
