# Description

yolo-pt: A PyTorch implementation for YOLO. Tested on YOLOv8.

Currently made to work with stricktly single class models.

# Requirements

opencv-python
torch (and its requirements)
numpy

May also need ultralytics depending on user system.

# Installation

`pip install git+https://github.com/vmc-7645/yolo-pt.git`

# Running

Prereq: Ensure you have a YOLO trained PyTorch model and test image ready.

```python

# run model on image
yolo.runpt(
    "images/test.png", 
    modelloc="weights/model.pt",
    sensitivity=0.6,
    overlap=0.3,
    img_show=True,
    upscale_img_mult=2
)

```