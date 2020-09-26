#!/bin/bash 
#PBS -l nodes=1:ppn=32:xe 
#PBS -l walltime=48:00:00 
#PBS -q normal 
#PBS -N testjob 
#PBS -e $PBS_JOBNAME$PBS_JOBID.err 
#PBS -o $PBS_JOBNAME$PBS_JOBID.out 
#PBS -m abe 
#PBS -M kshivam2@illinois.edu 
#PBS -l gres=shifter

SCRIPTBASEPATH=$PBS_O_WORKDIR 
SCRIPTNAME=$0 
PROGNAME="spiking_reservoir_rl_elastica" 
NTHREADS=32 NNODES=1

##########################################
#                                        #
#   Output some useful job information.  #
#                                        #
##########################################

# echo ------------------------------------------------------ 
# echo -n 'Job is running on node '; cat $PBS_NODEFILE 
# echo ------------------------------------------------------ 
# echo PBS: qsub is running on $PBS_O_HOST 
# echo PBS: originating queue is $PBS_O_QUEUE 
# echo PBS: executing queue is $PBS_QUEUE 
# echo PBS: working directory is $PBS_O_WORKDIR 
# echo PBS: execution mode is $PBS_ENVIRONMENT 
# echo PBS: job identifier is $PBS_JOBID 
# echo PBS: job name is $PBS_JOBNAME 
# echo PBS: node file is $PBS_NODEFILE 
# echo PBS: current home directory is $PBS_O_HOME 
# echo PBS: PATH = $PBS_O_PATH 
# echo ------------------------------------------------------ 

cd $PBS_O_WORKDIR

# Load modules
module load bwpy

# Run script
aprun -d 32 bwpy-environ run_python.sh
