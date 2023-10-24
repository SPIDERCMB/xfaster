#!/usr/bin/env bash
#SBATCH --job-name=xf_iter
#SBATCH --mem=6144
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:00:00
#SBATCH --nice=0

# Iterative run to calculate gcorr. latlon ps takes only
cd /home/sureng/xfaster/gcorr/
# python iterate.py --gcorr-config gcorr_configs/pointsource_latlon.ini &> submit/log_iterate.log
python iterate_nojobs.py --gcorr-config gcorr_configs/pointsource_latlon.ini &> submit/log_iterate.log