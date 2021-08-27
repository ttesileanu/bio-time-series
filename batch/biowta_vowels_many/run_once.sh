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
    --average-dwell 1500 \
    --min-dwell 800 \
    --rate-range 0.002 0.02 --rate-log \
    --exp-streak-range 2.0 100.0 --exp-streak-log \
    --temperature-range 1.0 100.0 --temperature-log \
    --timescale-range 10.0 150.0 --timescale-log \
    --economy \
    test_$2_$1.hdf5 vowel $2 biowta \
    > logs/$2_$1.out \
    2> logs/$2_$1.err
