#PBS -q normal
#PBS -S /bin/csh
#PBS -N reformat
# This example uses the Sandy Bridge nodes
# User job can access ~31 GB of memory per Sandy Bridge node.
# A memory intensive job that needs more than ~1.9 GB
# per process should use less than 16 cores per node
# to allow more memory per MPI process. This example
# asks for 32 nodes and 8 MPI processes per node.
# This request implies 32x8 = 256 MPI processes for the job.
#PBS -l select=143:ncpus=3:mpiprocs=3:model=bro
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -W group_list=g26209
#PBS -m e

# Load a compiler you use to build your executable, for example, comp-intel/2015.0.090.

module load mpi-hpe/mpt 
module load pkgsrc/2021Q2 
source /u/jwang23/miniconda3/bin/activate



# By default, PBS executes your job from your home directory.
# However, you can use the environment variable
# PBS_O_WORKDIR to change to the directory where
# you submitted your job.

cd /nobackup/jwang23/PODAAC/swot-crossover/

# use of dplace to pin processes to processors may improve performance
# Here you request to pin processes to processors 4-11 of each Sandy Bridge node.
# For other processor types, you may have to pin to different processors.

# The resource request of select=32 and mpiprocs=8 implies
# that you want to have 256 MPI processes in total.
# If this is correct, you can omit the -np 256 for mpiexec
# that you might have used before.

mpiexec -np 429  /u/jwang23/miniconda3/bin/python convert_netcdf_mpi.py >log 

# It is a good practice to write stderr and stdout to a file (ex: output)
# Otherwise, they will be written to the PBS stderr and stdout in /PBS/spool,
# which has limited amount  of space. When /PBS/spool is filled up, any job
# that tries to write to /PBS/spool will die.

# -end of script-
