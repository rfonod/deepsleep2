#!/bin/bash

###################################################################
#script_name	: download_cloud.sh
#description	: Downloads the '/training' and/or '/test' folders of the PhysioNet 2018 Challenge dataset from the dedicated GCP bucket (challenge-2018-1.0.0.physionet.org).
#author			: Robert Fonod
#version		: 1.1
#usage			: bash data/download_cloud.sh (files are saved next to this script, i.e. in data/)
#notes			: Requires the Google Cloud SDK ('gsutil')
#bash_version	: 3.2.57(1)-release
###################################################################

# download into the folder containing this script (data/), wherever it is called from
cd "$(dirname "$0")" || exit 1

function download_training () {
  echo "Downloading the 'training' folder ..."
  gsutil -m cp -r gs://challenge-2018-1.0.0.physionet.org/training .
}

function download_test () {
  echo "Downloading the 'test' folder ..."
  gsutil -m cp -r gs://challenge-2018-1.0.0.physionet.org/test .
}

function download_records () {
  echo "Downloading the 'RECORDS' file ..."
  gsutil cp gs://challenge-2018-1.0.0.physionet.org/RECORDS .
}

while true; do
  read -p "Do you want to download the 'training' folder from the 'gs://challenge-2018-1.0.0.physionet.org/' bucket (y/n)?" yn1
  case $yn1 in
  
    [Yy]* ) 
      break
      ;;
      
    [Nn]* ) 
      break
      ;;
      
    * ) 
      echo "Please answer 'yes' or 'no'"
      ;;
  esac
done

while true; do
  read -p "Do you want to download the 'test' folder from the gs://challenge-2018-1.0.0.physionet.org/ bucket (y/n)? Note: it has no arousal labels and is not used by the notebook." yn2
  case $yn2 in
  
    [Yy]* ) 
      break
      ;;
      
    [Nn]* ) 
      break
      ;;
      
    * ) 
      echo "Please answer 'yes' or 'no'"
      ;;
  esac
done

while true; do
  read -p "Do you want to download the 'RECORDS' file from the gs://challenge-2018-1.0.0.physionet.org/ bucket (y/n)?" yn3
  case $yn3 in
  
    [Yy]* ) 
      break
      ;;
      
    [Nn]* ) 
      break
      ;;
      
    * ) 
      echo "Please answer 'yes' or 'no'"
      ;;
  esac
done

case $yn1 in

  [Yy]* ) 
    download_training
    ;;
esac

case $yn2 in

  [Yy]* ) 
    download_test
    ;;
esac

case $yn3 in

  [Yy]* ) 
    download_records
    ;;
esac
