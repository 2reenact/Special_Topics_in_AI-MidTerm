# Special_Topics_in_AI_Final
### Name: 이제연
### SUID: 72210298
### Original Paper: ["UniverSeg: Universal Medical Image Segmentation"](http://arxiv.org/abs/2304.06131).
[Victor Ion Butoi](https://victorbutoi.github.io)\*,
[Jose Javier Gonzalez Ortiz](https://josejg.com)\*,
[Tianyu Ma](https://www.linkedin.com/in/tianyu-ma-472219174/),
[Mert R. Sabuncu](https://sabuncu.engineering.cornell.edu/),
[John Guttag](https://people.csail.mit.edu/guttag/),
[Adrian V. Dalca](http://www.mit.edu/~adalca/). (\*denotes equal contribution)
 
[![Explore UniverSeg in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10Vrbb6kyelXeGlGbmJhyhNkG9YwnwiBY)<br>

## 1. Introduction

![network](https://raw.githubusercontent.com/JJGO/UniverSeg/gh-pages/assets/images/network-architecture.png)

UniverSeg is an approach to learning a single general medical-image segmentation model that performs well on a variety of tasks without any retraining, including tasks that are substantially different from those seen at traning time.

Existing image segmentation models are perform poorly given out-of-distribution examples. For example, fine-tuning models trained on the natural image domain
can be unhelpful in the medical domain, likely due to the differences in data sizes, features, and task specifications between domains, and importantly still requires substantial retraining.

UniverSeg learn how to exploit an input set of labeled examples that specify the segmentation task, to segment a new biomedical image in one forward pass.

```python
# Set the encoder blocks using cross_conv
enc_blocks = []
for encoder_blocks do
  enc_blocks.append(CrossConv2D(encoder_blocks))

# Set the decoder blocks using cross_conv
dec_blocks = []
if decoder_blocks is None:
  decoder_blocks = encoder_blocks[::-1]
for decoder_blocks do
  dec_blocks.append(CrossConv2D(decoder_blocks))

# Convert the target image to a tensor(B, 1, H, W).
target = rearrange(target_image, "B 1 H W -> B 1 1 H W")
# Concat the support images and labels to a tensor(B, 2, H, W).
support = cat([support_images, support_labels], dim=2)

pass_through = []
for enc_blocks do
  target, support = enc_block(target, support)
    if encoder_block != last_encoder_block:
    # store the intermediate output in the pass_through list.
      pass_through.append((target, support))
  # Downsample the target and support tensors.
  target = downsample(target)
  support = downsample(support)

for decoder_blocks do
  target_skip, support_skip = pass_through.pop()
  # Upsample the target and support tensors.
  target = upsample(target, target_skip, dim=2)
  support = upsample(support, support_skip, dim=2)
  # Apply the decoder block to the target and support tensors.
  target, support = dec_block(target, support)

# Convert the target tensor to a tensor(B, C, H, W).
target = rearrange(target, "B 1 C H W -> B C H W")
# Apply the output convolution layer to the target tensor.
target = Conv2D(target)

return target
```

## 2. Installation
- **With pip**:
```shell
pip install git+https://github.com/2reenact/Special_Topics_in_AI_MidTerm.git
```

- **Manually**:
```shell
git clone https://github.com/2reenact/Special_Topics_in_AI_MidTerm
python -m pip install -r ./UniverSeg/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./UniverSeg)"
```

## 3. Run

The universeg architecture is described in the [`model.py`](https://github.com/2reenact/Special_Topics_in_AI_MidTerm/blob/master/universeg/model.py#L125) file.
Authors provide model weights a part of their [release](https://github.com/JJGO/UniverSeg/releases/tag/weights)).

To instantiate the UniverSeg model (and optionally use provided weights):
```python
from universeg import universeg

model = universeg(pretrained=True)

# To perform a prediction (where B=batch, S=support, H=height, W=width)
prediction = model(
    target_image,        # (B, 1, H, W)
    support_images,      # (B, S, 1, H, W)
    support_labels,      # (B, S, 1, H, W)
) # -> (B, 1, H, W)

```
For all inputs ensure that pixel values are min-max normalized to the $[0,1]$ range and that the spatial dimensions are $(H, W) = (128, 128)$.

