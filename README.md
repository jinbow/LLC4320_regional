# LLC4320_regional
A tool to convert the regional LLC4320 to netcdf files. Without modification, it only works within AMES supercomputer. 

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

From the cluster, run ```mpiexec -np ??? python convert_netcdf_mpi.py```. The ??? is the number of parallel processes you will use. You may need to change the settings in the main routine of the program. 


The output files are saved in the folder: ```/nobackup/jwang23/llc4320_stripe/regional.subsets.adac.netcdf```

Acknowledgment:

The json files for the MITgcm output are 'donated' by Ian Fenty. 
