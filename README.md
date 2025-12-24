# S2ID

Scale Invariant Image Diffuser

Uses relative positional coordinates (fourier series) added to the pixel "tokens", then uses axial attention
and text conditioning with an FFN in style of text transformer blocks, but adapted for images. Trained to predict
epsilon noise to be used as a diffusion model. Size and aspect ratio invariant, especially if trained with proper image
augmentations.

To train, run `train_v2.py`
To render the images, run `inference_v2.py`
To read the architecture code, look at `modules/s2id.py`