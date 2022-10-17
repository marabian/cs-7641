#!/bin/bash

export CLASSPATH=ABAGAIL.jar:$CLASSPATH

rm -rf data
mkdir -p data/csv
mkdir -p data/plot

# knapsack
echo "knapsack"
jython knapsack.py
python plot.py "knapsack"

# continuous peaks
echo "continuouspeaks"
jython continuouspeaks.py
python plot.py "continuouspeaks"

# # traveling salesman
echo "travelingsalesman"
jython travelingsalesman.py
python plot.py "travelingsalesman"