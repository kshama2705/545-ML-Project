import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image

import re
from itertools import combinations
import nltk
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

from nltk.corpus import wordnet
from model import *

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Load the Helper to Prepreocess Pascal Voc Dataset.
from helper import VOC2007Detection

virtexModel, imageLoader, sample_images, valid_subs = create_objects()
image_file = "redcaps/Pascal3.jpg"
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

def word2inds(word, caption, logitdict):
    from pdb import set_trace as bkpt 
    #bkpt()
    caption = ''.join(caption[:caption.find(word)].split())
    index = len(caption)
    count = 0
    ind = [i for i in range(len(logitdict)) if logitdict[i] == '://'][0] + 1
    while count < index:
        count += len(logitdict[ind])
        ind += 1
    result = []
    while count < index+len(word):
        result.append(ind)
        count += len(logitdict[ind])
        ind += 1
    return result

# Bounding Box Function
def createBbox(grayscale_cam, input_numpy):
  heat = (grayscale_cam >= 0.5)
  indices = np.where(heat == 1)
  cv2.imwrite("currentImage.jpg", input_numpy*255)
  boundingImage = cv2.imread("currentImage.jpg")
  start = (np.amin(indices[1]), np.amin(indices[0]))
  end = (np.amax(indices[1]),np.amax(indices[0]))
  print(start, end)
  color = (255, 0, 0)
  thickness = 2
  bbox = cv2.rectangle(boundingImage, start, end, color, thickness)
  return bbox, start, end

class VirtexModelOutputWrapper(torch.nn.Module):
    def __init__(self, model): 
        super(VirtexModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        tmp=self.model.predict_labels(x, sub_prompt='i took a picture', prompt='itap of a')
        self.caption = tmp[1]
        self.caption_dict=tmp[3]
        #return self.model.predict(x)[2].reshape(1,-1)
        return tmp[2].reshape(1,-1)

# Set the Manual Seed to force the model to generate the same caption
torch.manual_seed(10)
captioning_model = VirtexModelOutputWrapper(virtexModel)
virtexModel.model.decoder.nucleus_size = nuc_size

image = image.convert("RGB")

image_dict = imageLoader.transform(image)
image_tensor = image_dict['image']

coco_labels=['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

labels = ["cow", "dog","bird","Aeroplane",'bike','bicycle','car']
subreddit, caption, logits, logit2word = virtexModel.predict_labels(
            image_tensor, sub_prompt=sub, prompt='itap of a'
        )
potential_words = [virtexModel.tokenizer.decode([x.item()]) for x in torch.arange(logits.shape[1])[logits[0] > logit_threshold]]
potential_nouns = [extract_nouns("i took a picture of a " + word) for word in potential_words if extract_nouns("i took a picture of a " + word)]
# Matches potential Nouns in the generated caption to the potential pseudo label in the respective dataset class names. 
selected_nouns = [noun for noun in potential_nouns if max_word(noun, coco_labels)]
pseudo_label_to_caption_words=[max_word(noun, coco_labels) for noun in potential_nouns if max_word(noun, coco_labels)]

words_to_gradcam=selected_nouns 
word_tokens=[virtexModel.tokenizer.encode(x)[0] for x in words_to_gradcam]
#logits=[logits[0,x].item() for x in word_tokens]
#print(logits)

# GradCAM on the words 
gradcam_layer=[virtexModel.model.visual.cnn.layer4[-1]]
cam = GradCAM(model=captioning_model, target_layers=gradcam_layer)
for word_token in word_tokens:
    targets=[ClassifierOutputTarget(word_token)]
    #targets=[ClassifierOutputTarget(word_token) for word_token in word_tokens]
    #print(targets)
    grayscale_cam = cam(input_tensor=image_tensor,targets=targets)
    grayscale_cam = grayscale_cam[0, :] # First Index Correspondes to Batch Size
    #plt.imshow(grayscale_cam)
    #plt.show()
    input_numpy=image_tensor.numpy().astype('float32')
    input_numpy = (input_numpy -np.min(input_numpy))/(np.max(input_numpy)-np.min(input_numpy))
    input_numpy=input_numpy[0,:,:,:]
    input_numpy=np.transpose(input_numpy,(1,2,0))
        
    visualization = show_cam_on_image(input_numpy,grayscale_cam, use_rgb=True)
    superimposed_image=Image.fromarray(visualization)
    #superimposed_image.title(selected_nouns[i])
    superimposed_image.show()
    
    # Get the Bounding Box coordinates using the Bounding Box Function 
    bbox,start,end=createBbox(grayscale_cam,input_numpy)

    scale = max(224/image.size[0], 224/image.size[1])
    wshift = (scale*image.size[0] - 224)/2
    hshift = (scale*image.size[1] - 224)/2

    start = (int(round((start[0] + wshift)/scale)), int(round((start[1] + hshift)/scale)))
    end = (int(round((end[0] + wshift)/scale)), int(round((end[1] + hshift)/scale)))

    color = (255, 0, 0)
    thickness = 2
    bbox2 = cv2.rectangle(np.array(image), start, end, color, thickness)

    #print("Coordinates of the bounding box",start,end)
    plt.imshow(bbox)
    plt.show()
    plt.imshow(bbox2)
    plt.show()
print(selected_nouns)




