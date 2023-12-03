#!/bin/bash
##SBATCH --account=cs678fl23
#SBATCH --job-name=instructor
#SBATCH --qos=gpu
#SBATCH --partition=gpuq
#SBATCH -N 1
#SBATCH --ntasks-per-node=12
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --output=instructor.%j.out
#SBATCH --error=instructor.%j.err
#SBATCH --mem=200GB
#SBATCH --time=0-15:00:00



### Load modules
module load gnu10

### Activate virtual environment
source /home/$USER/miniconda/bin/activate
source activate instructor


### Execute program
python train.py --model_name_or_path sentence-transformers/use-cmlm-multilingual --output_dir outputs_cmlm_2e5augmented  --cache_dir train_cache/ --max_source_length 512 --num_train_epochs 1 --save_steps 5000 --cl_temperature 0.1 --warmup_ratio 0.1 --learning_rate 2e-5 --overwrite_output_dir