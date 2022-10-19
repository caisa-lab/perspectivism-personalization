#!/bin/bash
#

python ../src/ft_bert_no_verdicts.py \
--use_authors='false' \
--author_encoder='priming' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='sentence-transformers/all-distilroberta-v1' \
--bert_tok='sentence-transformers/all-distilroberta-v1' \
--sbert_dim=768 \
--user_dim=-1 \
--model_name='sbert' \
--split_type='author' \
--situation='title' \
--authors_embedding_path=''