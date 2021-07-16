# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

DATA_RNG=13746

./run_hyper.py \
    -n50 \
    --n-signals 200 \
    -d$DATA_RNG \
    -c$1 \
    -o$1 \
    --normalize \
    --max-pole-radius 0.95 \
    --rate-range 0.007 0.3 --rate-log \
    --exp-streak-range 1.0 1.0 \
    --cepstral-order-range 1 6 \
    --economy \
    test_$1.hdf5 3 0 cepstral \
    > logs/$1.out \
    2> logs/$1.err
