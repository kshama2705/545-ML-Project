import io
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from PIL import Image
from collections import defaultdict

import re
from itertools import combinations
import nltk
import os
import subprocess
import random
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

from nltk.corpus import wordnet
from model import *

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from torchvision.utils import draw_bounding_boxes
from torch.utils.data import DataLoader




# ----------------------------------------------------------------------------
def extract_nouns(img_caps):
    # extract nouns
    is_noun = lambda pos: True if pos[0] == 'N' else False
    pos = nltk.pos_tag(nltk.word_tokenize(img_caps))[-1]
    if is_noun(pos[1]):
        return pos[0]
    return None


def max_word(word, classes, max_word_threshold=0.4):
    nc = len(classes)
    similarity = np.zeros(nc)
    for i in range(nc):
        wordFromList1 = wordnet.synsets(classes[i])
        wordFromList2 = wordnet.synsets(word)
        if wordFromList1 and wordFromList2:
            similarity[i] = wordFromList1[0].wup_similarity(wordFromList2[0])

    max_class = classes[np.argmax(similarity)]

    if np.max(similarity) > max_word_threshold:
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
def createBbox(grayscale_cam, input_numpy, heat_threshold=0.5):
  heat = (grayscale_cam >= heat_threshold)
  indices = np.where(heat == 1)
  if len(indices[0]) == 0:
      return None, None, None
  cv2.imwrite("currentImage.jpg", input_numpy*255)
  boundingImage = cv2.imread("currentImage.jpg")
  start = (np.amin(indices[1]), np.amin(indices[0]))
  end = (np.amax(indices[1]), np.amax(indices[0]))
  #print(start, end)
  color = (255, 0, 0)
  thickness = 2
  bbox = cv2.rectangle(boundingImage, start, end, color, thickness)
  return bbox, start, end

def filter_bboxes(bboxes, intersection_threshold=0.5):
    if len(bboxes) < 2:
        return bboxes

    #Divide bboxes into groups based on iou
    groups = defaultdict(list)
    groups[0].append(bboxes[0])
    for i in range(1, len(bboxes)):
        matched = False
        for g in groups.keys():
            if bb_intersection_over_union(groups[g][0][1], bboxes[i][1]) > intersection_threshold:
                groups[g].append(bboxes[i])
                matched = True
                break
        if not matched:
            groups[len(groups)].append(bboxes[i])

    output = []
    #Average over each group
    for g in groups.keys():
        if len(groups[g]) > 1:
            newbox = [int(round(sum([x[1][i] for x in groups[g]])/len(groups[g]))) for i in range(4)]
            newconfidence = sum([x[0] for x in groups[g]])/len(groups[g])
            output.append((newconfidence, newbox))
        else:
            output.append(groups[g][0])
    return output

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou
    
def export_label(label, filepath):
    with open(filepath, 'w') as f:
        for obj in label['annotation']['object']:
            f.write(f"{obj['name'][0].lower().replace(' ', '')} {obj['bndbox']['xmin'][0]} {obj['bndbox']['ymin'][0]} {obj['bndbox']['xmax'][0]} {obj['bndbox']['ymax'][0]}{' difficult' if obj['difficult'][0] == 1 else ''}\n")

def export_pred(bboxes_flat, filepath):
    with open(filepath, 'w') as f:
        for label, confidence, bbox in bboxes_flat:
            f.write(f"{label.lower().replace(' ', '')} {confidence} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")



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

def eval_threshold(logit_threshold = -6, max_word_threshold = 0.4, heat_threshold = 0.5, intersection_threshold = 0.5):
    # Set the Manual Seed to force the model to generate the same caption
    torch.manual_seed(10)
    voc_instance=torchvision.datasets.VOCDetection(root="./redcaps/datasets",year='2012',download=True,transform=imageLoader.image_transform)
    #print(len(voc_instance))
    data = iter(DataLoader(voc_instance, batch_size=1, shuffle=False, num_workers=0))
    sub = 'i took a picture'
    coco_labels=['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    VOC_labels = ["__background__","Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car","Cat","Chair","Cow","Dining table","Dog","Horse","Motor bike","Person","Potted plant","Sheep","Sofa","Train","Tv monitor"]
    #Change here to switch between labels
    labels=VOC_labels
        
    for itr in range(100):
        image_tensor, label = next(data)
        #print(label)
        #image_file = "redcaps/datasets/Pascal3.jpg"
        #image = Image.open(image_file)
        #num_captions = 1

        #image = image.convert("RGB")
        #image_dict = imageLoader.transform(image)
        #image_tensor = image_dict
        subreddit, caption, logits, logit2word = virtexModel.predict_labels(
                    image_tensor, sub_prompt=sub, prompt='itap of a'
                )
        potential_words = [virtexModel.tokenizer.decode([x.item()]) for x in torch.arange(logits.shape[1])[logits[0] > logit_threshold]]
        confidence = torch.exp(logits[0, logits[0] > logit_threshold])
        potential_nouns = [(extract_nouns("i took a picture of a " + word), confidence) for word, confidence in zip(potential_words, confidence) if extract_nouns("i took a picture of a " + word)]
        potential_nouns, confidence = zip(*potential_nouns)
        # Matches potential Nouns in the generated caption to the potential pseudo label in the respective dataset class names. 
        selected_nouns = [noun for noun in potential_nouns if max_word(noun, labels, max_word_threshold)]
        pseudo_label_to_caption_words=[(max_word(noun, labels, max_word_threshold), confidence) for noun, confidence in zip(potential_nouns, confidence) if max_word(noun, labels, max_word_threshold)]
        pseudo_label_to_caption_words, confidence = zip(*pseudo_label_to_caption_words)
        if len(pseudo_label_to_caption_words) == 0:
            return
        
        words_to_gradcam=selected_nouns 
        word_tokens=[virtexModel.tokenizer.encode(x)[0] for x in words_to_gradcam]
        #logits=[logits[0,x].item() for x in word_tokens]
        #print(logits)

        # GradCAM on the words 
        gradcam_layer=[virtexModel.model.visual.cnn.layer4[-1]]
        cam = GradCAM(model=captioning_model, target_layers=gradcam_layer)
        bboxes = []

        for word_index,word_token in enumerate(word_tokens):
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
            #superimposed_image.show()
            
            # Get the Bounding Box coordinates using the Bounding Box Function 
            bbox,start,end=createBbox(grayscale_cam,input_numpy,heat_threshold)
            if bbox is None:
                bboxes.append(None)
                continue

            wit,hig = int(label['annotation']['size']['width'][0]), int(label['annotation']['size']['height'][0])
            scale = max(224/wit, 224/hig)
            wshift = (scale*wit - 224)/2
            hshift = (scale*hig - 224)/2

            start = (int(round((start[0] + wshift)/scale)), int(round((start[1] + hshift)/scale)))
            end = (int(round((end[0] + wshift)/scale)), int(round((end[1] + hshift)/scale)))

            color = (255, 0, 0)
            thickness = 2
            #bbox2 = cv2.rectangle(np.array(image), start, end, color, thickness)
            bboxes.append(list(start) + list(end))

            #print("Coordinates of the bounding box",start,end)
            #plt.imshow(bbox)
            #plt.show()
            #plt.imshow(bbox2)
            #plt.show()
        #print(selected_nouns)

        bbox_by_label = defaultdict(list)
        for box, key, confidence in zip(bboxes, pseudo_label_to_caption_words, confidence):
            if box is not None:
                bbox_by_label[key].append((confidence, box))

        # Remove duplicates
        for key in bbox_by_label.keys():
            bbox_by_label[key] = filter_bboxes(bbox_by_label[key], intersection_threshold)

        #print(bbox_by_label)



        bboxes_flat = []
        for key in bbox_by_label.keys():
            for conf, bbox in bbox_by_label[key]:
                bboxes_flat.append((key, conf, bbox))


        fname = label['annotation']['filename'][0]
        fname = fname[:fname.rfind('.')]
        filepath=f"./redcaps/mAP/input/detection-results/{fname}.txt"
        gtfilepath=f"./redcaps/mAP/input/ground-truth/{fname}.txt"
        if not os.path.exists('./redcaps/mAP/input/detection-results/'):
            os.makedirs('./redcaps/mAP/input/detection-results/')
        if not os.path.exists('./redcaps/mAP/input/ground-truth/'):
            os.makedirs('./redcaps/mAP/input/ground-truth/')

        export_pred(bboxes_flat, filepath)
        export_label(label, gtfilepath)

        print("The Process has been done or one image")

    fname = f'exp_{hash((logit_threshold, max_word_threshold, heat_threshold, intersection_threshold))}.txt'
    fname2 = fname.replace('.txt', '.pt')
    torch.save((logit_threshold, max_word_threshold, heat_threshold, intersection_threshold), fname2)
    cmd = ['python', './redcaps/mAP/main.py', '-np', '-na']
    subprocess.Popen(cmd, stdout=open(fname, 'w')).wait()
    
    #plt.imshow(draw_bounding_boxes(torch.from_numpy(np.array(image)).permute(2,0,1), bboxes, labels).numpy())

nuc_size = 0.8
virtexModel, imageLoader, sample_images, valid_subs = create_objects()
virtexModel.model.decoder.nucleus_size = nuc_size
captioning_model = VirtexModelOutputWrapper(virtexModel)



#logit_threshold = -6, max_word_threshold = 0.4, heat_threshold = 0.5, intersection_threshold = 0.5
for exps in range(100):
    logit_threshold = 4*random.random()-8           #[-8,-4]
    max_word_threshold = random.random()*0.8+0.2    #[0.2,1.0]
    heat_threshold = random.random()*0.8+0.2        #[0.2,1.0]
    intersection_threshold = random.random()*0.8+0.2 #[0.2,1.0]
    eval_threshold(logit_threshold, max_word_threshold, heat_threshold, intersection_threshold)

