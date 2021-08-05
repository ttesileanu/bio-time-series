module purge

module load slurm
module load disBatch

sbatch -p ccn,gen -n 160 --ntasks-per-node 40 disBatch task_file.db
