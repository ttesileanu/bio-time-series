# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

DATA_RNG=1942

./run_hyper_snippets.py \
    -n50 \
    --n-features 3 \
    --n-signals 200 \
    -d$DATA_RNG \
    -c$1 \
    -o$1 \
    --normalize \
    --rate-range 0.00005 0.005 --rate-log \
    --exp-streak-range 5.0 20.0 --exp-streak-log \
    --n-features-range 20 50 \
    --economy \
    test_$1.hdf5 vowel ao xcorr \
    > logs/$1.out \
    2> logs/$1.err
