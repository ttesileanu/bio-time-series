# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

./run_hyper.py \
    -n100 \
    --n-signals 50 \
    --n-features 3 \
    -d$1 \
    -c$1 \
    -o$1 \
    --normalize \
    --max-pole-radius 0.95 \
    --rate-range 0.007 0.3 --rate-log \
    --exp-streak-range 1.0 1.0 \
    --cepstral-order-range 1 6 \
    --economy \
    test_$1.hdf5 3 3 cepstral \
    > logs/$1.out \
    2> logs/$1.err
