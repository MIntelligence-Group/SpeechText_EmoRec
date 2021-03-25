# TOWARDS THE EXPLAINABILITY OF MULTIMODAL SPEECH EMOTION RECOGNITION 

## Note:
1. The code files are currently private as the corresponding research paper in InterSpeech'21 is under review. They will be made publically available soon after the paper is published/accepted for publication.
2. The corresponding paper, architecture diagrams and supplementary material containing the detailed results (Embedding plots, Intersection Matrices and Confusion Matrices) will also be shared later.

## To Reproduce the results for:
### IEMOCAP Dataset
   Run all the the three notebooks in the following order:
   1. Data_Preprocess(IEMOCAP).ipynb
   2. Training(IEMOCAP).ipynb
   3. Analysis(IEMOCAP).ipynb
   
### MSP-IMPROV Dataset
   Run all the the three notebooks in the following order:
   1. Data_Preprocess(IMPROV).ipynb
   2. Training(IMPROV).ipynb
   3. Analysis(IMPROV).ipynb

### RAVDESS Dataset
  To start with the RAVDESS data, you have to run all the the three notebooks in the following order:
   1. Data_Preprocess(RAVDESS).ipynb
   2. Training(RAVDESS).ipynb
   3. Analysis(RAVDESS).ipynb

## Additional Requirements
For IEMOCAP code
   1. nltk==3.5
   2. 6B token Glove Embedding

For RAVDESS code
   1. SoundFile==0.9.0
   2. numpy==1.17.1
   3. librosa==0.4.2
   4. glob3==0.0.1
   5. matplotlib==3.0.3
   6. seaborn==0.9.1
   7. Keras==2.3.1
   8. sklearn   
