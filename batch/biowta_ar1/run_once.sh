# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

./run_hyper.py -n500 --n-signals 25 -d$1 -c$1 -o$1 --rate-range 0.00001 0.10 --exp-streak-range 1.0 7.5 --rate-log test_$1.hdf5 1 0 > logs/$1.out 2> logs/$1.err
