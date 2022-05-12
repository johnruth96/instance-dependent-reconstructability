#!/bin/bash

DATASET_NAME=$1
DEGREE=$2
TECHNIQUE=$3
QUERIES=$4

# https://stackoverflow.com/questions/13910087/shell-script-to-capture-process-id-and-kill-it-if-exist

BLAZEGRAPH_DIR=lib/blazegraph
BLAZEGRAPH_BIN=blazegraph.jar
PROJECT_ROOT=$(pwd)
PYTHON_SCRIPT=$(realpath src/thesis/experiments/run.py)
GRAPH_FILE=$(realpath experiments/graphs/${DATASET_NAME}_${DEGREE}_${TECHNIQUE}.ttl)

# Function
get_pid() {
  echo $(ps -ef | grep ${BLAZEGRAPH_BIN} | grep -v grep | awk '{print $2}')
}

# Remove old BlazeGraph Instance
set -ex
rm -f ${BLAZEGRAPH_DIR}/blazegraph.jnl
if [ $(get_pid) ]; then
  kill $(get_pid)
fi

# Load Data
cd ${BLAZEGRAPH_DIR} &&
  java -cp ${BLAZEGRAPH_BIN} com.bigdata.rdf.store.DataLoader props.properties "${GRAPH_FILE}"

# Start BlazeGraph
java -jar ${BLAZEGRAPH_BIN} &
sleep 10

# Run experiment
cd "${PROJECT_ROOT}" &&
  python "${PYTHON_SCRIPT}" --time --add-precision --dataset ${DATASET_NAME} --degree ${DEGREE} --technique ${TECHNIQUE} --queries ${QUERIES}

# Shut down BlazeGraph
if [ $(get_pid) ]; then
  kill $(get_pid)
fi
