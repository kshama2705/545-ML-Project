import io

import streamlit as st
from model import *

# # TODO:
# - Reformat the model introduction
# - Make the iterative text generation


def gen_show_caption(sub_prompt=None, cap_prompt=""):
    with st.spinner("Generating Caption"):
        subreddit, caption = virtexModel.predict(
            image_dict, sub_prompt=sub_prompt, prompt=cap_prompt
        )
        st.markdown(
            f"""
            <style>
                red {{ color:#c62828; font-size: 1.5rem }}
                blue {{ color:#2a72d5; font-size: 1.5rem }}
                remaining {{ color: black; font-size: 1.5rem }}
            </style>

            <red>r/{subreddit}</red>: <blue> {cap_prompt} </blue><remaining> {caption} </remaining>
            """,
            unsafe_allow_html=True,
        )

with st.spinner("Loading Model"):
    virtexModel, imageLoader, sample_images, valid_subs = create_objects()


# ----------------------------------------------------------------------------
# Populate sidebar.
# ----------------------------------------------------------------------------
select_idx = None

st.sidebar.title("Select or upload an image")
if st.sidebar.button("Random Sample Image"):
    select_idx = get_rand_idx(sample_images)

sample_image = sample_images[0 if select_idx is None else select_idx]


uploaded_image = None
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    uploaded_image = Image.open(io.BytesIO(uploaded_file.getvalue()))
    select_idx = None  # Set this to help rewrite the cache


st.sidebar.title("Select a Subreddit")
sub = st.sidebar.selectbox(
    "Type below to condition on a subreddit. Select None for a predicted subreddit",
    valid_subs,
)

st.sidebar.title("Write a Custom Prompt")
cap_prompt = st.sidebar.text_input("Write the start of your caption below", value="")

_ = st.sidebar.button("Regenerate Caption")


st.sidebar.title("Advanced Options")
num_captions = st.sidebar.select_slider(
    "Number of Captions to Predict", options=[1, 2, 3, 4, 5], value=1
)
nuc_size = st.sidebar.slider(
    "Nucleus Size:\nLarger values lead to more diverse captions",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05,
)
# ----------------------------------------------------------------------------

virtexModel.model.decoder.nucleus_size = nuc_size

image_file = sample_image

# LOAD AND CACHE THE IMAGE
if uploaded_image is not None:
    image = uploaded_image
elif select_idx is None and "image" in st.session_state:
    image = st.session_state["image"]
else:
    image = Image.open(image_file)

image = image.convert("RGB")

st.session_state["image"] = image


image_dict = imageLoader.transform(image)

show_image = imageLoader.show_resize(image)

st.title("Image Captioning with VirTex model trained on RedCaps")
st.markdown("""
Caption your own images or try out some of our sample images.
You can also generate captions as if they are from specific subreddits,
as if they start with a particular prompt, or even both.
Tweet your results with `#redcaps`!

**Note:** This model was not trained on images of people,
hence may not generate accurate captions describing humans.
For more details, visit [redcaps.xyz](https://redcaps.xyz) check out
our [NeurIPS 2021 paper](https://openreview.net/forum?id=VjJxBi1p9zh).
""")

_, center, _ = st.columns([1, 10, 1])

with center:
    st.image(show_image)

    if sub is None and imageLoader.text_transform(cap_prompt) != "":
        st.write("Without a specified subreddit we default to /r/pics")
    for i in range(num_captions):
        gen_show_caption(sub, imageLoader.text_transform(cap_prompt))
