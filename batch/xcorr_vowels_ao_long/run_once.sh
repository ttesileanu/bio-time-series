# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

DATA_RNG=1940

./run_hyper_snippets.py \
    -n50 \
    --n-signals 200 \
    -d$DATA_RNG \
    -c$1 \
    -o$1 \
    --normalize \
    --average-dwell 1500 \
    --min-dwell 800 \
    --rate-range 0.00001 0.05 --rate-log \
    --exp-streak-range 3.0 150.0 --exp-streak-log \
    --economy \
    test_$1.hdf5 vowel ao xcorr \
    > logs/$1.out \
    2> logs/$1.err
