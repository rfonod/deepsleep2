#!/bin/bash

###################################################################
#script_name	: download_web.sh
#description	: Downloads the '/training' and/or '/test' folders of the PhysioNet 2018 Challenge dataset from web: https://physionet.org/content/challenge-2018/1.0.0/
#author			: Robert Fonod
#version		: 1.0
#usage			: bash download_web.sh | sh download_web.sh | ./download_web.sh
#notes			: The script uses 'wget' in a pseudo-parallel fashion
#bash_version	: 3.2.57(1)-release
###################################################################

function download_training () {
  for i in {3..14} 
  do 
    wget -r -N -np -b -nH --cut-dirs=3 -R "index.html*" -A "tr0$i*" https://physionet.org/files/challenge-2018/1.0.0/training/
  done
  echo "Downloading (in the background) the 'training' folder ..."
}

function download_test () {
  for i in {3..14} 
  do 
    wget -r -N -np -b -nH --cut-dirs=3 -R "index.html*" -A "te0$i*" https://physionet.org/files/challenge-2018/1.0.0/test/
  done
  echo "Downloading (in the background) the 'test' folder ..."
}

while true; do
  read -p "Do you want to download the 'training' folder from PhysioNet (y/n)?" yn1
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
  read -p "Do you want to download the 'test' folder from PhysioNet (y/n)?" yn2
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


