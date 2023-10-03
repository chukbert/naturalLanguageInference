# Natural Language Inference for Bahasa Indonesia Using pytorch

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model](#model)
4. [Word Embeddings](#word-embeddings)
5. [Training Results](#training-results)
6. [Usage](#usage)
7. [Challenges and Future Work](#challenges-and-future-work)
8. [References](#references)

## Introduction
Welcome to the "Natural Language Inference for Bahasa Indonesia" project! This project focuses on a Natural Language Processing (NLP) task known as Natural Language Inference (NLI) specifically designed for the Bahasa Indonesia language. NLI involves determining the relationship between pairs of sentences, typically categorized as entailment, contradiction, or neutral. In this project, we leverage the IndoNLI Dataset, which is a valuable resource for NLI tasks in Bahasa Indonesia. The result notebook [`NLIIndonesia.ipynb`](https://github.com/chukbert/naturalLanguageInference/blob/master/NLIIndonesia.ipynb).

## Dataset
The dataset used for this project is the IndoNLI Dataset, which can be downloaded from [this link](https://github.com/ir-nlp-csui/indonli). It consists of over 18,000 sentence pairs, with more than 12,000 pairs for training and 5,000 pairs for testing. Each sentence pair is labeled with its corresponding relationship (entailment, contradiction, or neutral), making it suitable for NLI model training and evaluation.

## Model
We employ the "Decomposable Attention Model," which was originally introduced by [Parikh et al. (2016)](https://paperswithcode.com/paper/a-decomposable-attention-model-for-natural) for NLI tasks. This model consists of three main stages: attending, comparing, and aggregating implemented using **pytorch** from scratch. It has proven effective in capturing complex relationships between sentence pairs, making it a suitable choice for our Bahasa Indonesia NLI task. You can find the original paper detailing this model [here](https://paperswithcode.com/paper/a-decomposable-attention-model-for-natural).

## Word Embeddings
For word embeddings, we utilize pretrained word embeddings from fastText specifically trained for the Bahasa Indonesia language. You can download these embeddings from the [fastText website](https://fasttext.cc/docs/en/crawl-vectors.html). Pretrained embeddings help our model understand the semantics and contextual information of Bahasa Indonesia words, enhancing its performance. Download pretrained vector embeddings and place it in the `pretrained` directory.

## Training Results
After training the Decomposable Attention Model on the IndoNLI Dataset, we obtained the following results:
![alt text](https://github.com/chukbert/naturalLanguageInference/blob/master/img/output.svg "result")
- Training Loss: 0.367
- Training Accuracy: 0.833
- Test Accuracy: 0.449

It's important to note that these results, while a valuable step forward, are relatively lower compared to models trained on much larger English NLI datasets. This discrepancy is expected, given the significant difference in dataset size, with English NLI datasets often exceeding 550,000 labeled sentence pairs. Nonetheless, this project represents a crucial milestone in advancing NLP for Bahasa Indonesia, as NLI applications are diverse and demand extensive research in this field.

## Usage
```
conda install --file requirements.txt
```
## Challenges and Future Work
The challenges encountered in this project, such as limited dataset size and language-specific nuances, highlight the need for further research and resource development in Bahasa Indonesia NLP. Future work can include:
- Expansion of the IndoNLI Dataset.
- Exploring more advanced NLI architectures.
- Fine-tuning models for specific NLI subtasks.

## References and Citation
- Original Paper on Decomposable Attention Model: [Link](https://paperswithcode.com/paper/a-decomposable-attention-model-for-natural)
- IndoNLI Dataset: [Link](https://github.com/ir-nlp-csui/indonli)
```
@inproceedings{mahendra-etal-2021-indonli,
    title = "{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
    author = "Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.821",
    pages = "10511--10527",
}
```
- Pretrained Word Embeddings for Bahasa Indonesia: [Link](https://fasttext.cc/docs/en/crawl-vectors.html)
```
@inproceedings{grave2018learning,
  title={Learning Word Vectors for 157 Languages},
  author={Grave, Edouard and Bojanowski, Piotr and Gupta, Prakhar and Joulin, Armand and Mikolov, Tomas},
  booktitle={Proceedings of the International Conference on Language Resources and Evaluation (LREC 2018)},
  year={2018}
}
```
Feel free to explore the code and resources in this repository to gain a deeper understanding of our NLI project for Bahasa Indonesia. If you have any questions or feedback, please don't hesitate to reach out. Thank you for your interest in our NLP research!
