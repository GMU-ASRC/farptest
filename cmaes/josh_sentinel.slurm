#!/bin/bash

## Give your job a name to distinguish it from other jobs you run.
#SBATCH --job-name=jhudd_sentinel_01

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=normal

## Separate output and error messages into 2 files.`
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID`
#SBATCH --output=/scratch/%u/%x-%N-%j.out  # Output file`
#SBATCH --error=/scratch/%u/%x-%N-%j.err   # Error file`

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=mabayneh@gmu.edu     # Put your GMU email address here

#SBATCH --nodes=1  ##number of nodes to use
#SBATCH --ntasks=1  ##number of tasks to run

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=5GB        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=0-02:00  # Total time needed for job: Days-Hours:Minutes:Seconds

#SBATCH --cpus-per-task 48


## Commands to Load Modules:
# module load use.own uv

cd ~/cmaes_stuff/jhudd_farptest/ || { echo "Unable to navigate to farptest"; exit 1; }
source .venv/bin/activate || { echo "Unable to activate venv"; exit 1; }
cd cmaes/ || { echo "Unable to cd into 'cmaes'"; exit 1; }

## Load modules, insert code, and run your programs here.
python sentinel_cma.py -n 6 -rs 32 -t 1000 -p 10 -mi 50