#!/bin/bash
set -x

cancel_slurm_job() {
jobid=$(squeue -o "%A %j" -p llm2 -u $USER | grep $GITHUB_RUN_ID-$GITHUB_JOB | awk '{print $1}')
if [ -n "$jobid" ];then
   echo "The job $jobid will be canceled"
   scancel $jobid
   sleep 0.1
   cancel_slurm_job   
else
   echo "There has been no job needed canceled"
fi
}

cancel_slurm_job

exit 0
