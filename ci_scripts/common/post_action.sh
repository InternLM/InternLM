#!/bin/bash
set -x

readonly JOB_LOG=$GITHUB_WORKSPACE/${GITHUB_JOB}.log

if [[ ! -f ${JOB_LOG} ]]; then
   echo "There is no ${JOB_LOG}. May be there is no job needed canceled"
   exit 0
fi

jobid=$(grep "queued and waiting" ${JOB_LOG} | grep -oP "\d+")
datetime=$(date '+%Y-%m-%d %H:%M:%S')
echo "$datetime,The slurm job $jobid will be canceled"
scancel $jobid
# double cancel
scancel $jobid
rm -rf ${JOB_LOG}
exit 0
