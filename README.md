
# Reproduction of: One Embedder, Any Task: Instruction-Finetuned Text Embeddings

This repository contains the code and pre-trained models for CS 678 Project Submission.

Please refer to https://github.com/xlang-ai/instructor-embedding for original research. This repository has been forked from the author's repository.

To run the evaluation on the 4 models generated for Project Checkpoint 1, please use the file ***Instructor_Reproduction.ipynb*** provided in the repo.

Please refer to ***Instructor_training.ipynb*** to train another checkpoint using different hyperparameters.

Please refer to ***Error_Analysis.ipynb*** to run analysis on sentences logged during evaluation. All sentences logged during evaluation have a non matching predicted label from the test corpus. The sentences will need to be manually provided in sentence_a to evaluate. You can use sentence_b as is. This contains one sentence per label to run pair classification with sentences in sentence_a array.




