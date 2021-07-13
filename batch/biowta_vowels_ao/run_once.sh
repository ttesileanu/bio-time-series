# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

./run_hyper_snippets.py \
    -n100 \
    --n-signals 50 \
    -d$1 \
    -c$1 \
    -o$1 \
    --normalize \
    --rate-range 0.001 0.1 --rate-log \
    --exp-streak-range 2.0 10.0 \
    --temperature-range 0.0 2.0 \
    --timescale-range 1.0 4.0 --timescale-log \
    --economy \
    test_$1.hdf5 vowel ao biowta \
    > logs/$1.out \
    2> logs/$1.err
