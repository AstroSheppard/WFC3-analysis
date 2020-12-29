#!/bin/sh

# First: Switch into correct virtual env
# This script will run the data reduction routines

planet=$1
visit=$2
transit=$3
xw=$4
if [ $# -eq 6 ]
then
    plotting=$5
else
    plotting=0
fi

python2 bkg.py $planet $visit $xw
python2 reduction.py $planet $visit $transit $plotting
