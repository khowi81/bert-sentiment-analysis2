Bert-Sentiment-Analysis: Fine Tuning BERT for Sentiment Analysis on Tweets (Sentiment 140 dataset)

Overview: This project fine tunes a pretrained BERT model (BERT-base-uncased) on a sample of 100k tweets from the Sentiment 140 dataset. 
   The goal was to build an end to end ML workflow, from preprocessing and tokenization to model training, evaluation and experiment
   tracking with Weights and Biases. I just wanted to try my hand at ML in a production setting, so I ran it on 3 different platforms: Colab, Databricks and my own GPU.
   
Tech Stack: 
  Hugging Face Transformers and datasets
  PyTorch
  Weights and Biases (tracking)
  Colab
  local NVIDIA RTX 4060 GPU (2nd run of the model)
  Databricks (3rd run of model)
  MLflow (tracking, 3rd run of model)
  
Workflow: 
 Load dataset samples from Sentiment 140 dataset
 Preprocess labels (map 0/4 to binary 0/1)
 Tokenize text with HuggingFace Tokenizer
 Fine tune BERT (3 weights, AdamW optimizer, weight decay regularization)
 Track results with wandb (see screenshots)

 Results: 
 <img width="1864" height="753" alt="wandb_logs" src="https://github.com/user-attachments/assets/97c19513-efce-4106-ada7-b0358516a64b" />

  graphs included show the very first run on Colab. Not bad. I ran out of GPU time on Colab, so I will add more complete graphs later. Note, using the NVIDIA RTX 4060 only took 1 hour, whereas the free tier on Colab took 4 hours. 
  

Future Work: 
 Continue to experiment with larger subsets of the dataset
 Continue to work with local GPU, as the free tier on Colab isn't sufficient
 Continue running model locally to determine accuracy of model and tuning parameters. 

 How to run the model: 
git clone https://github.com/khowi81/bert-sentiment-analysis.git
cd bert-sentiment-analysis
pip install -r requirements.txt
jupyter notebook notebook/sentiment_bert.ipynb


 Special Note: the Jupyter notebook has some notes, as I was learning as I was building. 

 ## ✨ Acknowledgments
- Hugging Face community  
- Sentiment140 dataset creators  
- Weekend hackathon spirit ✌️ 
