#!/bin/bash
#SBATCH --job-name=getSpeciesScale    # Job name
#SBATCH --mail-type=BEGIN,FAIL,END               # Mail events (NONE, BEGIN, END, FAIL, AL$
#SBATCH --mail-user=sergio.marconi@weecology.org   # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b   

#SBATCH --ntasks=1                 # Number of MPI ranks
#SBATCH --cpus-per-task=1            # Number of cores per MPI rank 

#SBATCH --mem=4000
#SBATCH --time=90:00:00       #Time limit hrs:min:sec
#SBATCH --output=getSp.out   # Standard output and error log
#SBATCH --error getSp.err


pwd; hostname; date
ml conda

source activate quetzal3
echo "getSp"
python3 scale.py $1

echo "done"

date

