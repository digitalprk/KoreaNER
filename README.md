# KoreaNER / 한국어 개체명

---

**Evaluation & Comparison:**

Corpus: National Institute of Korean Language (ROK) - NER Corpus / 국립국어원 - 개체명 인식용 말뭉치[18]

| Category      | KoNER/코너 (2016)             | Annie (2016)                 | KoreaNER                     | 
| ------------- |:----------------------------:|:----------------------------:|:----------------------------:|
|               | Precision | Recall | F-Score | Precision | Recall | F-Score | Precision | Recall | F-Score |
| DT            | 0.894     | 0.880  | 0.887   | 0.6373    | 0.7785 | 0.7009  | 0.96      | 0.81   | 0.88    |
| LC            | 0.793     | 0.853  | 0.822   | 0.5822    | 0.8782 | 0.7002  | 0.71      | 0.73   | 0.72    |
| OG            | 0.824     | 0.772  | 0.797   | 0.7624    | 0.7087 | 0.7346  | 0.70      | 0.66   | 0.68    |
| PS            | 0.915     | 0.885  | 0.899   | 0.8834    | 0.6127 | 0.7236  | 0.84      | 0.73   | 0.78    |
| TI            | 0.872     | 0.810  | 0.840   | 0.5441    | 0.8810 | 0.6727  | 0.96      | 0.81   | 0.88    |

<sub><sup>[Source][17]</sub></sup>

---

**Future improvements:**

- Add Jamo (Korean base character) level embeddings
- Add POS information

---

**References:**

[Character-Aware Neural Language Models][2]
[Boosting Named Entity Recognition with Neural Character Embeddings][1]
[Attending To Characters In Neural Sequence Labeling Models][3]

[Neural Architectures for Named Entity Recognition][4]

[Bidirectional LSTM-CRF Models for Sequence Tagging][5]

[Character-level Convolutional Networks for Text Classification][6]

[A Syllable-based Technique for Word Embeddings of Korean Words][16]

**Open source projects (Github):**

[CharCNN Pytorch][7]

[word2vec-keras-in-gensim][8]

[anago][9]

[annie][10]

[DeepSequenceClassification][11]

[autumn_ner][12]

[kchar][13]

[deep-named-entity-recognition][14]

[Sequence Tagging with Tensorflow][15]

  [1]: https://arxiv.org/pdf/1505.05008.pdf
  [2]: https://arxiv.org/pdf/1505.05008.pdf
  [3]: https://aclweb.org/anthology/C/C16/C16-1030.pdf
  [4]: https://arxiv.org/pdf/1603.01360.pdf
  [5]: https://arxiv.org/pdf/1508.01991.pdf
  [6]: https://arxiv.org/pdf/1509.01626.pdf
  [7]: https://github.com/srviest/char-cnn-pytorch
  [8]: https://github.com/SimonPavlik/word2vec-keras-in-gensim/blob/keras106/word2veckeras
  [9]: https://github.com/Hironsan/anago/blob/master/anago
  [10]: https://github.com/krikit/annie/tree/master/bin
  [11]: https://github.com/napsternxg/DeepSequenceClassification/
  [12]: https://github.com/tttr222/autumn_ner/blob/master/model.py
  [13]: https://github.com/jarfo/kchar/
  [14]: https://github.com/aatkinson-old/deep-named-entity-recognition/blob/master/
  [15]: https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
  [16]: https://arxiv.org/pdf/1708.01766.pdf
  [17]: https://www.korean.go.kr/common/download.do;front=6F1EBE4CC5ED872C7FEB60294C6E1D8C?file_path=etcData&c_file_name=c00c0198-220a-47a5-bccd-55eda862dbe4_0.pdf&o_file_name=2016%EB%85%84%20%EA%B5%AD%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%EC%A0%95%EB%B3%B4%20%EC%8B%9C%EC%8A%A4%ED%85%9C%20%EA%B2%BD%EC%A7%84%20%EB%8C%80%ED%9A%8C%20%EB%B0%9C%ED%91%9C%20%EC%9E%90%EB%A3%8C%EC%A7%91.pdf
  [18]: https://ithub.korean.go.kr/

