for datafile in "$@"
do
  filename="${datafile##*/}"
  echo $filename
  sbatch batchSpeciesScale.SLURM $filename
done
