#!/bin/bash
set -x

retry_times=3
for ((i=1;i<=$retry_times;i++));do
    jobid=$(squeue -o "%A %j" -u $USER | grep ${GITHUB_RUN_ID}-${GITHUB_JOB} | awk '{print $1}')
    if [[ -n "$jobid" ]];then
        echo "The job $jobid will be canceled."
        scancel $jobid
        sleep 0.5
    else
        echo "There are no more jobs that need to be canceled."
        break
    fi
done

if [[ $i -gt $retry_times ]];then
    echo "There have been tried $retry_times times. Please contact user $USER to confirm the job status."
fi

exit 0
