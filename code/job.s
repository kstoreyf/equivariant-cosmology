#!/bin/bash
##SBATCH --job-name=gen_configs
##SBATCH --job-name=halo_table
#SBATCH --job-name=mah_table_try2
##SBATCH --job-name=halo_selector_tng100_Mmin10.25
##SBATCH --job-name=geo_featurizer_tng100_table
##SBATCH --job-name=scalar_featurizer_tng100_table
##SBATCH --job-name=feature_importance_tng100_nstarmin10_mstellar
##SBATCH --job-name=train_nn_multi_scalars_Mmin10.25_yerr0.05
##SBATCH --job-name=train_nn_m_stellar_catalog_z0_list_nl9
##SBATCH --job-name=train_nn_num_mergers_geos
##SBATCH --job-name=train_nn_j_stellar
##SBATCH --job-name=compute_mrv
##SBATCH --job-name=train_nn_Mofa_epochs1000_lr1e-3_hs128_scalars
##SBATCH --job-name=train_nn_a_mfrac_39_epochs2000_lr5e-5_hs128_scalars_gx1_gv1_n5
##SBATCH --job-name=feature_info_MI
#SBATCH --output=logs/%x.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=60GB
#SBATCH --time=48:00:00

# need somewhere >30 and <60 GB for train_nn.py
# now only need 10GB for train_nn!
# for scalar featurizer, >10 and <60
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
#python halo_table.py;
python mah_tables.py;
#python run_halo_selector.py;
#python run_geometric_featurizer.py;
#python run_scalar_featurizer.py;
#python feature_importance.py;
#python train_nn.py
#python feature_info.py
#python compute_halo_properties.py
"



