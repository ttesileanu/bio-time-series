# !/bin/bash

# run_once <rng_seed> <use_temp> <use_streak> <use_timescale> <suffix>

rng_seed=$1
use_temp=$2
use_streak=$3
use_timescale=$4
suffix=$5

n_models=2
ar_order=3
ma_order=0
min_dwell=50
avg_dwell=100

if [ ! $use_temp -eq 0 ]
then
    temp_range="0.0 3.0"
else
    temp_range="0.0 0.0"
fi
if [ ! $use_streak -eq 0 ]
then
    streak_range="2.0 12.0"
else
    streak_range="2.0 2.0"
fi
if [ ! $use_timescale -eq 0 ]
then
    timescale_range="1.0 6.0"
else
    timescale_range="1.0 1.0"
fi

export OPENBLAS_NUM_THREADS=1

module load gcc
module load python3

source env/bin/activate

DATA_RNG=13746

./run_hyper.py \
    -n50 \
    --n-models $n_models \
    --n-signals 200 \
    -d$DATA_RNG \
    -c$rng_seed \
    -o$rng_seed \
    --normalize \
    --max-pole-radius 0.95 \
    --average-dwell $avg_dwell \
    --min-dwell $min_dwell \
    --rate-range 5e-5 2e-2 --rate-log \
    --exp-streak-range $streak_range \
    --temperature-range $temp_range \
    --timescale-range $timescale_range --timescale-log \
    --economy \
    test_$suffix.hdf5 \
    $ar_order \
    $ma_order \
    biowta \
    > logs/$suffix.out \
    2> logs/$suffix.err
