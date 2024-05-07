#!/bin/bash
#SBATCH --job-name=cvgckblkz
#SBATCH --partition=etna
#SBATCH --account=nano
#SBATCH --qos=normal
#SBATCH --exclusive=user
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time=96:00:00

module unload intel/2016.4.072
module load intel/2018.5.274.par
module load vasp_intelmpi/5.4.4.16052018

EXE="vasp_std"
time mpirun $EXE

rm CHG EIGENVAL REPORT PCDAT XDATCAR IBZKPT OSZICAR PROCAR

exit 0

