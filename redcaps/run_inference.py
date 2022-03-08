import io

from model import *

virtexModel, imageLoader, sample_images, valid_subs = create_objects()
image_file = '2.png'
image = Image.open(image_file)

sub = None

num_captions = 1
nuc_size = 0.8
# ----------------------------------------------------------------------------

virtexModel.model.decoder.nucleus_size = nuc_size

image = image.convert("RGB")

image_dict = imageLoader.transform(image)

subreddit, caption, logits, logit2word = virtexModel.predict(
            image_dict, sub_prompt=sub, prompt=''
        )
print(subreddit, caption, logits.shape, logit2word)