#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This scripts shutdowns idle Compute Engine instance.

# Number of sequential checks when the instance had utilization below the threshold.
COUNTER=0
# If actual CPU utilization is below this threshold script will increase the counter.
THRESHOLD_PERCENT=2
# Interval between checks of the CPU utilizations.
SLEEP_INTERVAL_SECONDS=5
# How big COUNTER need to be for the script to shutdown the instance. For example,
# if we want an instance to be stopped after 20min of idle. Each utilization probe
# happens every 5sec (SLEEP_INTERVAL_SECONDS), therefore since there are 1200 seconds
# in 20 min (20 * 60 = 1200) we need counter threshold to be 240 (1200 / 5).
HALT_THRESHOLD=240
CONTAINER_NAME="" # If using mutiple containers define CONTAINER_NAME value.


# `--check-docker` will only check Docker container idle CPU.
#  Modify ashutdown service to: ExecStart=/usr/local/bin/ashutdown --check-docker
if [[ $* == *--check-docker* ]]; then
  CHECK_DOCKER="true"
  if [ -x "$(docker -v)" ]; then
    echo 'Error: docker is not installed.' >&2
    exit 1;
  fi
  NUM_CONTAINERS=$(docker ps | tail -n +2 | wc -l)
  if [ "$NUM_CONTAINERS" -eq "0" ]; then
    echo "No Running containers"
    exit 1;
  fi
else
  CHECK_DOCKER="false"
fi


function err() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $@" >&2
  exit 1;
}


function check_cpu(){
  # Check CPU idle value for local VM or Docker container
  while true; do
   if [[ "$CHECK_DOCKER" = "true" ]]; then
    CPU_PERCENT=$(docker ps -f name="$CONTAINER_NAME" -q | xargs docker stats --format "{{.CPUPerc}}" --no-stream | \
    awk '{ print $1+0 }')
   else
    # Selects only the final lines (the ones starting with Average: and, among those, only select those whose
    # second column is numeric. For those lines, the third column (cpu load) is printed.
    # For multicore AI Notebooks we select CPU_PERCENT for the core with most CPU usage
    CPU_PERCENT=$(mpstat -P ALL 1 1 | awk '/Average:/ && $2 ~ /[0-9]/ {print $3}' | awk '{{max=0};if($1>max) {max=$1}} END {print max}')
   fi
   if (( $(echo "${CPU_PERCENT} < ${THRESHOLD_PERCENT}" | bc -l) )); then
     COUNTER=$((COUNTER + 1))
     if (( $(echo "${COUNTER} > ${HALT_THRESHOLD}" | bc -l) )); then
       shutdown now
     fi
   else
     COUNTER=0
   fi
   sleep "${SLEEP_INTERVAL_SECONDS}"
  done
}


main(){
  check_cpu || err "Error checking CPU"
}

main