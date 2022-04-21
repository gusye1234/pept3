#!/usr/bin/bash

DIR="./percolator_result"
if [ ! -d "$DIR" ]; then
  # Take action if $DIR exists. #
  echo "Creating ${DIR}..."
  mkdir $DIR
fi
percolator -v 0 --weights percolator_result/weights.csv \
                --post-processing-tdc --only-psms --testFDR 0.1 \
                --results-psms percolator_result/target.psms \
                --decoy-results-psms percolator_result/decoy.psms \
                ./demo_data/demo_out.tab