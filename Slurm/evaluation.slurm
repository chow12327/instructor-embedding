
#!/bin/bash
##SBATCH --account=cs678fl23
#SBATCH --job-name=instructor
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --error=instructor.%j.err
#SBATCH --mem=50GB
#SBATCH --time=0-2:30:00



### Load modules
module load gnu10

### Activate virtual environment
source /home/$USER/miniconda/bin/activate
source activate instructor


### Execute program
### Execute program
python evaluation/MTEB/examples/evaluate_model.py --model_name 'chow12327/instructor_cmlm_wiki_multilingual' --output_dir outputs --task_name 'EmotionClassification' --result_file results --batch_size 32 --samples_per_label 8 --n_experiments 5  --seed 30 --robustnessSamples 10