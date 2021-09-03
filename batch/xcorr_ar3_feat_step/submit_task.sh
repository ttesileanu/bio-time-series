module purge

module load slurm
module load disBatch

sbatch -p ccn,gen --constraint=skylake -n 40 --ntasks-per-node 40 disBatch task_file.db
