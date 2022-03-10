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
def max_word(wordlist, classes):
    nw = len(wordlist)
    nc = len(classes)
    similarity = np.zeros((nc, nw))
    for i in range(nc):
        for j in range(nw):
            wordFromList1 = wordnet.synsets(classes[i])
            wordFromList2 = wordnet.synsets(wordlist[j])
            if wordFromList1 and wordFromList2:
                similarity[i, j] = wordFromList1[0].wup_similarity(wordFromList2[0])

    similarity = np.sum(similarity, axis=1)
    return classes[np.argmax(similarity)]


# assuming we have a dataframe 'nouns' where rows are samples
# and columns are the nouns in each caption for that sample
def extract_targets(nouns, classes, method=2):
    # pick words that have highest similarity to every other target
    # if method is 1 extract 1 target per caption returns nxm
    #              2 extract 1 target per image returns nx1
    if method==1:
        return nouns.applymap(lambda x: max_word(re.split('\s+', x), classes))
    else:
        return nouns.apply(lambda row: max_word(re.split('\s+', ' '.join(row.values.astype(str))), classes), axis=1)


coco_names = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
              'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
              'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
              'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
              'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
              'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
              'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']