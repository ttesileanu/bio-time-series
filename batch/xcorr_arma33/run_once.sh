# !/bin/bash
export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

./run_hyper.py -n100 --n-signals 25 -d$1 -c$1 -o$1 --rate-range 3e-4 0.3 --exp-streak-range 2.5 7.5 --rate-log --economy test_$1.hdf5 3 3 xcorr > logs/$1.out 2> logs/$1.err
