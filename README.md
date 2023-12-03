
# Reproduction of: One Embedder, Any Task: Instruction-Finetuned Text Embeddings

This repository contains the code and pre-trained models for CS 678 Project Submission.

Please refer to https://github.com/xlang-ai/instructor-embedding for original research. This repository has been forked from the author's repository.

Please refer to ***Instructor_training.ipynb*** to train another a model using various hyperparameters.

To run the evaluation on the models generated for Project Checkpoint 1, please use the file ***Instructor_Reproduction.ipynb*** provided in the repo. Provide model directory name in --model_name argument.

Please refer to ***Error_Analysis.ipynb*** to run analysis on sentences logged during evaluation. All sentences logged during evaluation have a non matching predicted label from the test corpus. The sentences will need to be manually provided in sentence_a to evaluate. You can use sentence_b as is. This contains one sentence per label to run pair classification with sentences in sentence_a array.

Sample Slurm scipts for training and evaluation are available in Slurm directory.

Use our trained models from hugging face as follows:

from sklearn.metrics.pairwise import cosine_similarity
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('chow12327/instructor_cmlm_wiki_multilingual')

sentences_a = [['represent a sentence: ','set a reminder tomorrow for Steve to call Alex about a ride to the pool on saturday']]
               
sentences_b = [['represent a sentence: ','set a reminder  tomorrow for jodie to call karen about a ride to the pool on saturday'],
               ['represent a sentence: ','can you enable video call']]

embeddings_a = model.encode(sentences_a)
embeddings_b = model.encode(sentences_b)
similarities_ab = cosine_similarity(embeddings_a,embeddings_b)

print(similarities_ab)

The below models are available:
chow12327/instructor_cmlm_wiki_multilingual - CMLM multilingual model trained on multi-lingual MEDI Dataset
chow12327/instructor_cmlm_wiki - CMLM multilingual model trained on English MEDI Dataset
chow12327/instructor_wikionly - GTR-T5 model trained on English MEDI Dataset


