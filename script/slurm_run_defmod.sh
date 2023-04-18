#!/bin/sh
#
#SBATCH --job-name="Rundefmod"
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=research-ceg-gse

export PETSC_DIR=/home/jingmingruan/petsc
export PETSC_ARCH=dblue-ompi-gcc-2022r2
module load 2022r2
module load openmpi

# cd /scratch/jingmingruan/3D/3D_mfvaryoffset/intersection_angle/run_001mpa
# cd /scratch/jingmingruan/3D/3D_Xflt/45_vo
pwd
srun /home/jingmingruan/defmod-build/run/defmod-swpc/bin/defmod -f Buijze3D.inp -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor -fv 3 -ss 1

# srun /home/jingmingruan/defmod-build/origin/defmod-swpc/bin/defmod -f F3DX14_dsp.inp -pc_type lu -pc_factor_mat_solver_type mumps -ksp_monitor -ksp_error_if_not_converged

module load miniconda3/4.12.0
module load openssh
module load git

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# conda activate py3

# python Buijze3D_mf_mp.py
# python /home/jingmingruan/dataprocessing/fe_sort.py -f Buijze3D -dmn 3 -slip 1 -dyn 1 -rsf 0 -poro 1 -nobs 0 -seis 0

echo "End..."
