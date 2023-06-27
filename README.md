# Unifying Data Perspectivism and Personalization: An Application to Social Norms
Codebase for EMNLP22 work Unifying Data Perspectivism and Personalization: An Application to Social Norms and NLP+CSS work Understanding Interpersonal Conflict Types and their Impact on Perception Classification

## Dataset
You can download our dataset [here](https://drive.google.com/drive/folders/18iGMBEsQYw8dya9baqhrquQHmpmk71ka).

## Running 

* Create user attribution embeddings <br>
```python
python attribution_training.py --split_type='verdicts' --path_to_data='data/'
```

* Extract user embeddings
  
  1. Extract average embeddings:
     ```python
      python users_sbert_embeddings.py --path_to_data='data/' --dirname='data/path_to_history_comments/' --output_dir='data/embeddings/'
     ```
  2. Extract user attribution embeddings:
     ```python
      python users_attribution_embeddings.py --checkpoint_dir='verdicts_linear_layers_2022-06-16_12:12:57:866975.pt' --embedding_type='prediction'
     ```
* Find the scripts to run models [here](https://github.com/caisa-lab/perspectivism-personalization/tree/master/scripts)


## Citation 
If our code or models aided your research, please cite our [paper](https://aclanthology.org/2022.emnlp-main.500/) and if you used the models or data for conflict types please cite our [NLP+CSS paper](https://aclanthology.org/2022.nlpcss-1.10):
```bibtex
@inproceedings{plepi-etal-2022-unifying,
    title = "Unifying Data Perspectivism and Personalization: An Application to Social Norms",
    author = "Plepi, Joan  and
      Neuendorf, B{\'e}la  and
      Flek, Lucie  and
      Welch, Charles",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.500",
    pages = "7391--7402",
    abstract = "Instead of using a single ground truth for language processing tasks, several recent studies have examined how to represent and predict the labels of the set of annotators. However, often little or no information about annotators is known, or the set of annotators is small. In this work, we examine a corpus of social media posts about conflict from a set of 13k annotators and 210k judgements of social norms. We provide a novel experimental setup that applies personalization methods to the modeling of annotators and compare their effectiveness for predicting the perception of social norms. We further provide an analysis of performance across subsets of social situations that vary by the closeness of the relationship between parties in conflict, and assess where personalization helps the most.",
}
```
```bibtex
@inproceedings{welch-etal-2022-understanding,
    title = "Understanding Interpersonal Conflict Types and their Impact on Perception Classification",
    author = "Welch, Charles  and
      Plepi, Joan  and
      Neuendorf, B{\'e}la  and
      Flek, Lucie",
    booktitle = "Proceedings of the Fifth Workshop on Natural Language Processing and Computational Social Science (NLP+CSS)",
    month = nov,
    year = "2022",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nlpcss-1.10",
    pages = "79--88",
    abstract = "Studies on interpersonal conflict have a long history and contain many suggestions for conflict typology. We use this as the basis of a novel annotation scheme and release a new dataset of situations and conflict aspect annotations. We then build a classifier to predict whether someone will perceive the actions of one individual as right or wrong in a given situation. Our analyses include conflict aspects, but also generated clusters, which are human validated, and show differences in conflict content based on the relationship of participants to the author. Our findings have important implications for understanding conflict and social norms.",
}
```
