from collections import defaultdict
import glob
import json
import os
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from virtex.data import transforms as T

class ZeroShotDataset(Dataset):
    def __init__(
        self,
        data_root: str = "datasets/inaturalist",
        split: str = "train",
        image_transform: Callable = T.DEFAULT_IMAGE_TRANSFORM,
        label_map: str = None,
        tokenizer = None,
        model_dataset = 'redcaps',
        prompt_cls_sos = None,
        prompt_sos_eos = None
        ):
        
        self.data_root = data_root
        self.split = split
        self.label_map = json.load(open(label_map))
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.model_dataset = model_dataset
        self.prompt_cls_sos = prompt_cls_sos
        self.prompt_sos_eos = prompt_sos_eos

        im_id = 0
        
        self.image_id_to_file_path = {}
        self.instances = []
        
        for folder_name,labelname in self.label_map.items():
            image_folder = self.data_root + self.split + folder_name + "/"
            for image_file in [x for x in os.listdir(image_folder) if x[-4:]=='.jpg']:
                path = image_folder + image_file
                self.image_id_to_file_path[im_id] = path
                self.instances.append((im_id,labelname[1]))
                im_id+=1
                
                
        im_net_list =  [x[0].replace('_',' ').lower() for x in sorted(self.label_map.values(),key=lambda x: x[1])]
        
        print(im_net_list)
        
        cls_token = [tokenizer.token_to_id("[CLS]")]
        sos_token = [tokenizer.token_to_id("[SOS]")]
        eos_token =[tokenizer.token_to_id("[EOS]")]
        
        a_an_dets = [ " an " if cat[0].lower() in ["a","e","i","o","u"] else " a " for cat in im_net_list ]
        imagenet_tensors = [cls_token
                            +tokenizer.encode("i took a picture")
                            +sos_token
                            +tokenizer.encode("itap of "+a_an_dets[i]+im_net_list[i])
                            +eos_token 
                            for i in range(len(im_net_list))]

        imagenet_tensors_backward = [cls_token
                                     +tokenizer.encode("i took a picture")
                                     +eos_token
                                    +tokenizer.encode("itap of "+a_an_dets[i]+im_net_list[i])[::-1]
                                    +sos_token 
                                    for i in range(len(im_net_list))]
      

        tensor_lengths = torch.tensor([len(x) for x in imagenet_tensors])
        imagenet_tensors_forward = [torch.tensor(x) for x in imagenet_tensors]
        imagenet_tensors_backward = [torch.tensor(x) for x in imagenet_tensors_backward]    
        imagenet_tensors_forward = pad_sequence(imagenet_tensors_forward,batch_first=True)
        imagenet_tensors_backward = pad_sequence(imagenet_tensors_backward,batch_first=True)


        print("imagenet_tensors_forward.shape: ", imagenet_tensors_forward.shape)
        print("imagenet_tensors_backward.shape: ", imagenet_tensors_backward.shape)
        print("tensor_lengths.shape: ", tensor_lengths.shape)
         
        self.imagenet_tensors_forward = imagenet_tensors_forward
        self.imagenet_tensors_backward = imagenet_tensors_backward
        self.tensor_lengths = tensor_lengths.long()

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx: int):
        
        image_id, label = self.instances[idx]
        image_path = self.image_id_to_file_path[image_id]
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.image_transform(image=image)["image"]
            image = np.transpose(image, (2, 0, 1))
        except:
            print("$#%@#$%#image_path$@%:",image_path)
            image = np.random.rand(234, 325, 3)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.image_transform(image=image)["image"]
            image = np.transpose(image, (2, 0, 1))

        return {
            "image": torch.tensor(image, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "caption_tokens": self.imagenet_tensors_forward,
            "noitpac_tokens": self.imagenet_tensors_backward,
            "caption_lengths": self.tensor_lengths
        }
    
    @staticmethod
    def collate_fn(data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "image": torch.stack([d["image"] for d in data], dim=0),
            "label": torch.stack([d["label"] for d in data], dim=0),
            "caption_tokens": data[0]['caption_tokens'],
            "noitpac_tokens": data[0]['noitpac_tokens'],
            "caption_lengths": data[0]['caption_lengths']
        }