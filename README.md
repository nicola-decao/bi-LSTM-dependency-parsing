# Bidirectional LSTM for dependency parsing
### Disjoint predictions and complete classification accuracy in automated dependency parsing

A common approach for solving the dependency parsing problem is to apply the SHIFT-REDUCE algorithm in combination with neural networks outputting the desired transition and/or label at each iteration [1][2]. This study compares the performances of different models for labeled dependency parsing.

First, an unlabeled dependency parsing was implemented which consists of a bi-LSTM and an MLP on the top of it outputting the selected transition. This model was then extended by adding a two hidden layer MLP which takes the representations of the head and the tail of the transition and outputs one of 49 labels. This MLP was then altered to additionally accept as an input the parent of the current head. It was also built an other version that accepts the corresponding GloVe [3] word embeddings instead of LSTM output vectors. Finally it was created an architecture with a bi-LSTM followed by only one MLP, that predicts one of 99 possible labeled transitions out.

The purpose of this work is to evaluate such different architectures.

See ![report](https://github.com/nicola-decao/BILSTMDP/blob/master/report.pdf) for further details.

### Notes

Additional files are needed to run the project:
+ GloVe dataset to put into ```/data``` (download here http://nlp.stanford.edu/projects/glove/)
+ a training and a test files to put into ```/data```(download not available due to copyright)

To run the project simply run ```python main.py```.

### References

[1] James Cross and Liang Huang. 2016. Incremental
parsing with minimal features using bi-directional
LSTM. CoRR, abs/1606.06406.

[2] Eliyahu Kiperwasser and Yoav Goldberg. 2016. Sim-
ple and accurate dependency parsing using bidi-
rectional LSTM feature representations. CoRR,
abs/1603.04351.

[3] Jeffrey Pennington, Richard Socher, and Christo-
pher D. Manning. Glove: Global vectors for word
representation.
