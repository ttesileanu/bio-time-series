# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

./run_hyper.py \
    -n100 \
    --n-signals 50 \
    -d$1 \
    -c$1 \
    -o$1 \
    --normalize \
    --max-pole-radius 0.95 \
    --rate-range 0.0001 0.1 --rate-log \
    --exp-streak-range 1.0 15.0 \
    --economy \
    test_$1.hdf5 1 0 xcorr \
    > logs/$1.out \
    2> logs/$1.err
