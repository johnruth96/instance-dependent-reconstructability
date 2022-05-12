#!/bin/bash

while read line; do
  IFS=' ' read -a data <<<"$line"

  dataset=${data[0]}
  degree=${data[1]}
  technique=${data[2]}
  queries=${data[3]}

  if [[ $* == *--run* ]]; then
    echo "Running $dataset, $degree, $technique: $queries..."
    scripts/run.sh $dataset $degree $technique $queries
  else
    echo "Dry-run $dataset, $degree, $technique: $queries ..."
  fi
done <"$1"
