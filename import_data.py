import json


def load_corpus(file = 'NEtaggedCorpus_train.json'):
    with open(file, 'r', encoding = 'utf-8') as fp:
        corpus = json.load(fp)
    
    tagged_sentences = []
    for sentence in corpus['sentence']:
        current_sentence = [[a['lemma'], 'O'] for a in sentence['morp']]
        for named_entity in sentence['NE']:
            NE_begin = named_entity['begin']
            NE_end = named_entity['end']
            NE_type = named_entity['type']
            for i in range(NE_begin, NE_end + 1):
                current_sentence[i][1] = NE_type
        tagged_sentences.append(current_sentence)
    X = [[_[0] for _ in tagged_sentence] for tagged_sentence in tagged_sentences]
    y = [[_[1] for _ in tagged_sentence] for tagged_sentence in tagged_sentences]
    return X, y
        
    
