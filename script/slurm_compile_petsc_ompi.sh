#!/bin/sh
#
#SBATCH --job-name="compile_petsc"
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --account=research-ceg-gse

export PETSC_DIR=/home/jingmingruan/petsc
export PETSC_ARCH=dblue-ompi-gcc-2022r2
module load 2022r2
module load openmpi

cd petsc

./configure --with-mpi=1 --with-cc=mpicc --with-fc=mpif90 --with-cxx=mpicxx --download-fblaslapack --download-hdf5 --download-hdf5-fortran-bindings --download-scalapack --download-parmetis --download-mumps --download-metis --download-cmake --with-shared-libraries=0 --with-debugging=0 COPTFLAGS=-O3 FOPTFLAGS=-O3 CXXOPTFLAGS=-O3  


make all check
