#!/bin/bash
#

python ../src/ft_bert_with_users.py \
--use_authors='false' \
--author_encoder='none' \
--loss_type='focal' \
--num_epochs=10 \
--sbert_model='' \
--bert_tok='' \
--sbert_dim=768 \
--user_dim=-1 \
--model_name='judge_bert' \
--split_type='sit' \
--authors_embedding_path='' 



