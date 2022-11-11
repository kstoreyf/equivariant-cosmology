#!/bin/bash
##SBATCH --job-name=halo_selector_tng100
##SBATCH --job-name=geo_scalar_featurizer_tng50
##SBATCH --job-name=scalar_featurizer_tng100_nstarmin50_twin_pseudo_rall_mord2_n01
##SBATCH --job-name=feature_importance_tng100_nstarmin10_mstellar
#SBATCH --job-name=train_nn_m_stellar_epochs1000_lr5e-5_hs128_mrv
##SBATCH --job-name=train_nn_Mofa_epochs1000_lr0.0001_hs128
##SBATCH --job-name=train_nn_a_mfrac_n39_epochs1000_lr0.0001_hs128
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60GB
#SBATCH --time=24:00:00

# need somewhere >30 and <60 GB for train_nn.py
cd ~
overlay_ext3=/scratch/ksf293/overlay-50G-10M.ext3
singularity \
exec --overlay $overlay_ext3:ro \
/scratch/work/public/singularity/centos-7.8.2003.sif /bin/bash \
-c "source /ext3/env.sh; \
/bin/bash; \
cd /home/ksf293/equivariant-cosmology/code; \
conda activate eqenv;
#python generate_configs.py;
#python run_halo_selector.py;
#python run_geometric_featurizer.py;
#python run_scalar_featurizer.py;
#python feature_importance.py;
python train_nn.py
"



