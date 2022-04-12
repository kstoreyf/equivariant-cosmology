#!/bin/bash
##SBATCH --job-name=halo_selector_tng100_nstarmin1
#SBATCH --job-name=geo_featurizer_tng100_nstarmin1_xminPE
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
/bin/bash; \
cd /home/ksf293/equivariant-cosmology/code; \
#python run_featurizer.py;
#python run_halo_selector.py;
python run_geometric_featurizer.py;
"



