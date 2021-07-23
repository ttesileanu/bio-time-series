# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

DATA_RNG=1942

./run_hyper_snippets.py \
    -n50 \
    --n-signals 200 \
    -d$DATA_RNG \
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