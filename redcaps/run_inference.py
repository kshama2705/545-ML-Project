import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import re
from itertools import combinations
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.corpus import wordnet
from model import *

virtexModel, imageLoader, sample_images, valid_subs = create_objects()
image_file = "Pascal1.jpg"
image = Image.open(image_file)

sub = 'i took a picture'

num_captions = 1
nuc_size = 0.8
logit_threshold = -7
# ----------------------------------------------------------------------------
def extract_nouns(img_caps):
    # extract nouns
    is_noun = lambda pos: True if pos[0] == 'N' else False
    pos = nltk.pos_tag(nltk.word_tokenize(img_caps))[-1]
    if is_noun(pos[1]):
        return pos[0]
    return None


def max_word(word, classes):
    nc = len(classes)
    similarity = np.zeros(nc)
    for i in range(nc):
        wordFromList1 = wordnet.synsets(classes[i])
        wordFromList2 = wordnet.synsets(word)
        if wordFromList1 and wordFromList2:
            similarity[i] = wordFromList1[0].wup_similarity(wordFromList2[0])

    max_class = classes[np.argmax(similarity)]

    if np.max(similarity) > 0.4:
        return max_class

    return None

virtexModel.model.decoder.nucleus_size = nuc_size

image = image.convert("RGB")

image_dict = imageLoader.transform(image)

coco_labels=['__background__', 'person', 'bike', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

labels = ["cow", "dog","bird","Aeroplane",'bike','bicycle','car']
for i in range(5):
    subreddit, caption, logits, logit2word = virtexModel.predict_labels(
                image_dict, coco_labels, sub_prompt=sub, prompt='itap of a'
            )
    potential_words = [virtexModel.tokenizer.decode([x.item()]) for x in torch.arange(logits.shape[1])[logits[0] > logit_threshold]]
    potential_nouns = [extract_nouns("i took a picture of a " + word) for word in potential_words if extract_nouns("i took a picture of a " + word)]

    
print(potential_words)
print(potential_nouns)
print([max_word(noun, coco_labels)+"/"+noun for noun in potential_nouns if max_word(noun, coco_labels)])