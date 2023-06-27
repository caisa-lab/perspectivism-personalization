# Unifying Data Perspectivism and Personalization: An Application to Social Norms
Codebase for EMNLP22 work Unifying Data Perspectivism and Personalization: An Application to Social Norms

## Dataset
You can download our dataset [here](https://drive.google.com/drive/folders/18iGMBEsQYw8dya9baqhrquQHmpmk71ka).

## Running 

* Create user attribution embeddings <br>
`python attribution_training.py --split_type='verdicts' --path_to_data='data/'`

* Extract user embeddings
  
  1. Extract average embeddings: <br> `python users_sbert_embeddings.py --path_to_data='data/' --dirname='data/path_to_history_comments/' --output_dir='data/embeddings/'`
  2. Extract user attribution embeddings: <br> `python users_attribution_embeddings.py --checkpoint_dir='verdicts_linear_layers_2022-06-16_12:12:57:866975.pt' --embedding_type='prediction'` 
* Find the scripts to run models [here](https://github.com/caisa-lab/perspectivism-personalization/tree/master/scripts)
