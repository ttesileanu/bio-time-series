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
    --rate-range 0.0001 0.004 --rate-log \
    --exp-streak-range 1.0 3.0 --exp-streak-log \
    --economy \
    test_$1.hdf5 vowel ao xcorr \
    > logs/$1.out \
    2> logs/$1.err
