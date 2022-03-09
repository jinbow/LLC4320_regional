# LLC4320_regional
Routines to manipulate ECCO simulation 1/48 (LLC4320) with AMES supercomputers

Prerequist 

1. Install miniconda in the home directory
1. Include necessary modules
```
source miniconda3/bin/activate
module load pkgsrc/2021Q2 mpi-hpe/mpt
```
1. Run from interactive cluster
 1. Start an interactive cluster:
```qsub -I -q devel -lselect=$1:ncpus=$2:model=$3,walltime=2:00:00 ```
