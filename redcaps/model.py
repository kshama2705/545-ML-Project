import os
import json
import glob
import random
import torch
import torchvision

import wordsegment as ws
from PIL import Image
from huggingface_hub import hf_hub_url, cached_download

from virtex.config import Config
from virtex.factories import TokenizerFactory, PretrainingModelFactory
from virtex.utils.checkpointing import CheckpointManager

CONFIG_PATH = "/Users/vineet/Desktop/Winter -22 Courses/EECS 545/Project/545-ML-Project/redcaps/config.yaml"
MODEL_PATH = "redcaps/checkpoint.pth"
VALID_SUBREDDITS_PATH = "redcaps/subreddit_list.json"
SAMPLES_PATH = "./samples/*.jpg"


class ImageLoader:
    def __init__(self):
        self.image_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )
        self.show_size = 500

    def load(self, im_path):
        im = torch.FloatTensor(self.image_transform(Image.open(im_path))).unsqueeze(0)
        return {"image": im}

    def raw_load(self, im_path):
        im = torch.FloatTensor(Image.open(im_path))
        return {"image": im}

    def transform(self, image):
        im = torch.FloatTensor(self.image_transform(image)).unsqueeze(0)
        return {"image": im}

    def text_transform(self, text):
        # at present just lowercasing:
        return text.lower()

    def show_resize(self, image):
        # ugh we need to do this manually cuz this is pytorch==0.8 not 1.9 lol
        image = torchvision.transforms.functional.to_tensor(image)
        x, y = image.shape[-2:]
        ratio = float(self.show_size / max((x, y)))
        image = torchvision.transforms.functional.resize(
            image, [int(x * ratio), int(y * ratio)]
        )
        return torchvision.transforms.functional.to_pil_image(image)


class VirTexModel:

    def __init__(self):
        self.config = Config(CONFIG_PATH)
        ws.load()
        self.device = "cpu"
        self.tokenizer = TokenizerFactory.from_config(self.config)
        self.model = PretrainingModelFactory.from_config(self.config).to(self.device)
        CheckpointManager(model=self.model).load(MODEL_PATH)
        self.model.eval()
        self.valid_subs = json.load(open(VALID_SUBREDDITS_PATH))

    def predict(self, image_dict, sub_prompt=None, prompt=""):
        if sub_prompt is None:
            subreddit_tokens = torch.tensor(
                [self.model.sos_index], device=self.device
            ).long()
        else:
            subreddit_tokens = " ".join(ws.segment(ws.clean(sub_prompt)))
            subreddit_tokens = (
                [self.model.sos_index]
                + self.tokenizer.encode(subreddit_tokens)
                + [self.tokenizer.token_to_id("[SEP]")]
            )
            subreddit_tokens = torch.tensor(subreddit_tokens, device=self.device).long()

        if prompt != "":
            # at present prompts without subreddits will break without this change
            # TODO FIX
            cap_tokens = self.tokenizer.encode(prompt)
            cap_tokens = torch.tensor(cap_tokens, device=self.device).long()
            subreddit_tokens = (
                subreddit_tokens
                if sub_prompt is not None
                else torch.tensor(
                    (
                        [self.model.sos_index]
                        + self.tokenizer.encode("pics")
                        + [self.tokenizer.token_to_id("[SEP]")]
                    ),
                    device=self.device,
                ).long()
            )

            subreddit_tokens = torch.cat([subreddit_tokens, cap_tokens])

        is_valid_subreddit = False
        subreddit, rest_of_caption = "", ""
        image_dict["decode_prompt"] = subreddit_tokens
        while not is_valid_subreddit:

            #with torch.no_grad():
            output = self.model(image_dict)
            caption = output["predictions"][0].tolist()
            caption_logits = output['logits']
            caption_logits_full = output['logits_full']

            if self.tokenizer.token_to_id("[SEP]") in caption:
                sep_index = caption.index(self.tokenizer.token_to_id("[SEP]"))
                caption[sep_index] = self.tokenizer.token_to_id("://")

            logit2word = {}
            for i in range(len(caption)):
                logit2word[i] = self.tokenizer.decode(caption[:i + 1]).replace(self.tokenizer.decode(caption[:i]), '').strip()
            caption = self.tokenizer.decode(caption)

            if "://" in caption:
                subreddit, rest_of_caption = caption.split("://")
                subreddit = "".join(subreddit.split())
                rest_of_caption = rest_of_caption.strip()
            else:
                subreddit, rest_of_caption = "", caption.strip()

            # split prompt for coloring:
            if prompt != "":
                _, rest_of_caption = caption.split(prompt.strip())

            is_valid_subreddit = subreddit in self.valid_subs

        return subreddit, rest_of_caption, caption_logits, logit2word


    def predict_labels(self, image_dict, labels, sub_prompt=None, prompt=""):
        if sub_prompt is None:
            subreddit_tokens = torch.tensor(
                [self.model.sos_index], device=self.device
            ).long()
        else:
            subreddit_tokens = " ".join(ws.segment(ws.clean(sub_prompt)))
            subreddit_tokens = (
                [self.model.sos_index]
                + self.tokenizer.encode(subreddit_tokens)
                + [self.tokenizer.token_to_id("[SEP]")]
            )
            subreddit_tokens = torch.tensor(subreddit_tokens, device=self.device).long()

        if prompt != "":
            # at present prompts without subreddits will break without this change
            # TODO FIX
            cap_tokens = self.tokenizer.encode(prompt)
            cap_tokens = torch.tensor(cap_tokens, device=self.device).long()
            subreddit_tokens = (
                subreddit_tokens
                if sub_prompt is not None
                else torch.tensor(
                    (
                        [self.model.sos_index]
                        + self.tokenizer.encode("pics")
                        + [self.tokenizer.token_to_id("[SEP]")]
                    ),
                    device=self.device,
                ).long()
            )

            subreddit_tokens = torch.cat([subreddit_tokens, cap_tokens])

        is_valid_subreddit = False
        subreddit, rest_of_caption = "", ""
        image_dict["decode_prompt"] = subreddit_tokens

        label_tokens = [self.tokenizer.token_to_id(x) for x in labels]
        while not is_valid_subreddit:

            #with torch.no_grad():
            output = self.model(image_dict)
            caption = output["predictions"][0].tolist()
            caption_logits = output['logits']
            caption_logits_full = output['logits_full']

            if self.tokenizer.token_to_id("[SEP]") in caption:
                sep_index = caption.index(self.tokenizer.token_to_id("[SEP]"))
                caption[sep_index] = self.tokenizer.token_to_id("://")

            logit2word = {}
            for i in range(len(caption)):
                logit2word[i] = self.tokenizer.decode(caption[:i + 1]).replace(self.tokenizer.decode(caption[:i]), '').strip()
            caption = self.tokenizer.decode(caption)

            if "://" in caption:
                subreddit, rest_of_caption = caption.split("://")
                subreddit = "".join(subreddit.split())
                rest_of_caption = rest_of_caption.strip()
            else:
                subreddit, rest_of_caption = "", caption.strip()

            # split prompt for coloring:
            if prompt != "":
                rest_of_caption = caption.split(prompt.strip())

            is_valid_subreddit = subreddit in self.valid_subs

        return subreddit, rest_of_caption, caption_logits_full, logit2word


def download_files():
    # download model files
    download_files = [CONFIG_PATH, MODEL_PATH, VALID_SUBREDDITS_PATH]
    for f in download_files:
        fp = cached_download(hf_hub_url("zamborg/redcaps", filename=f))
        os.system(f"cp {fp} ./{f}")


def get_samples():
    return glob.glob(SAMPLES_PATH)


def get_rand_idx(samples):
    return random.randint(0, len(samples) - 1)


#@st.cache(allow_output_mutation=True)  # allow mutation to update nucleus size
def create_objects():
    sample_images = get_samples()
    virtexModel = VirTexModel()
    imageLoader = ImageLoader()
    valid_subs = json.load(open(VALID_SUBREDDITS_PATH))
    valid_subs.insert(0, None)
    return virtexModel, imageLoader, sample_images, valid_subs


footer = """<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>
*Please note that this model was explicitly not trained on images of people, and as a result is not designed to caption images with humans.

This demo accompanies our paper RedCaps.

Created by Karan Desai, Gaurav Kaul, Zubin Aysola, Justin Johnson
</p>
</div>
"""
