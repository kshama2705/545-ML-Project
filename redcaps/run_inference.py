import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import re
from itertools import combinations
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import stopwords, wordnet
from model import *

virtexModel, imageLoader, sample_images, valid_subs = create_objects()
image_file = "/Users/vineet/Desktop/Winter -22 Courses/EECS 545/Project/545-ML-Project/redcaps/Pascal1.jpg"
image = Image.open(image_file)

sub = 'i took a picture'

num_captions = 1
nuc_size = 0.8
logit_threshold = -7
# ----------------------------------------------------------------------------
def extract_nouns(img_caps):
    # load conventional stop words
    stop_words = stopwords.words('english')

    # remove stop words / can include removal of some nouns
    remove_stop = np.vectorize(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    img_caps = remove_stop(img_caps)
    print()
    print(str(img_caps))
    print()
    print(nltk.pos_tag(nltk.word_tokenize(str(img_caps))))

    # extract nouns
    is_noun = lambda pos: True if pos[0] == 'N' else False
    extract_nouns = np.vectorize(lambda x: ' '.join([word for (word, pos) 
                                                     in nltk.pos_tag(nltk.word_tokenize(x)) if is_noun(pos)]))
    return extract_nouns(img_caps)

virtexModel.model.decoder.nucleus_size = nuc_size

image = image.convert("RGB")

image_dict = imageLoader.transform(image)

coco_labels=['__background__', 'person', 'bike', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

labels = ["cow", "dog","bird","Aeroplane",'bike','bicycle','car']
for i in range(5):
    subreddit, caption, logits, logit2word = virtexModel.predict_labels(
                image_dict, coco_labels, sub_prompt=sub, prompt='itap of a'
            )
    print(subreddit, caption, logits.shape, logit2word)
    #print(logits[0])
    #print([coco_labels[x] for x in torch.argmax(logits, dim=1)][0])
    #print([virtexModel.tokenizer.decode([x.item()]) for x in torch.arange(logits.shape[1])[logits[0] > logit_threshold]])
    
    potential_words = [virtexModel.tokenizer.decode([x.item()]) for x in torch.arange(logits.shape[1])[logits[0] > logit_threshold]]
    potential_nouns = extract_nouns(" ".join(potential_words))
    print(potential_nouns)
    potential_labels = [] #TODO