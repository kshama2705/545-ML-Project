import re
import numpy as np
import pandas as pd
from itertools import combinations
import nltk
from nltk.corpus import stopwords, wordnet

# assuming we have a dataframe 'img_caps' where rows are samples
# and columns are the different captions for that sample
def extract_nouns(img_caps):
    # load conventional stop words
    stop_words = stopwords.words('english')

    # remove stop words / can include removal of some nouns
    remove_stop = np.vectorize(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    img_caps = img_caps.apply(remove_stop)

    # extract nouns
    is_noun = lambda pos: True if pos[0] == 'N' else False
    extract_nouns = np.vectorize(lambda x: ' '.join([word for (word, pos) 
                                                     in nltk.pos_tag(nltk.word_tokenize(x)) if is_noun(pos)]))
    return img_caps.apply(extract_nouns)

#
def max_word(wordlist):
    n = len(wordlist)
    similarity = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            wordFromList1 = wordnet.synsets(wordlist[i])
            wordFromList2 = wordnet.synsets(wordlist[j])
            if wordFromList1 and wordFromList2:
                similarity[i, j] = wordFromList1[0].wup_similarity(wordFromList2[0])

    similarity = np.sum(similarity, axis=1)
    return wordlist[np.argmax(similarity)]


# assuming we have a dataframe 'nouns' where rows are samples
# and columns are the nouns in each caption for that sample
def extract_targets(nouns, method=2):
    # pick words that have highest similarity to every other target
    # if method is 1 extract 1 target per caption returns nxm
    #              2 extract 1 target per image returns nx1
    if method==1:
        return nouns.applymap(lambda x: max_word(re.split('\s+', x)))
    else:
        return nouns.apply(lambda row: max_word(re.split('\s+', ' '.join(row.values.astype(str)))), axis=1)
