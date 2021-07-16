module purge

module load slurm
module load disBatch

# some parameters
start_seed=100
sims_per_model=40

# create task_file
rm -f task_file.db
n_tasks=0
for use_temp in {0..1}
do
    for use_streak in {0..1}
    do
        for use_timescale in {0..1}
        do
            crt_seed=${start_seed}
            for (( i = 0; i < $sims_per_model; i++ ))
            do
                echo "./run_once.sh ${crt_seed} ${use_temp} ${use_streak} ${use_timescale} mod${use_temp}${use_streak}${use_timescale}_${crt_seed}" >> task_file.db
                crt_seed=$(( $crt_seed + 1 ))
                n_tasks=$(( $n_tasks + 1 ))
            done
        done
    done
done

# run job
sbatch -p ccn,gen --constraint=skylake -n $n_tasks --ntasks-per-node 40 disBatch task_file.db
