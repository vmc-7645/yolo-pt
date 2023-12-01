# Description

yolo-pt: A PyTorch implementation for YOLO. Tested on YOLOv8.

Currently made to work with stricktly single class models.

Understandably, the methods used in this make the whole point of YOLO (only looking once, that is) kind of useless. That said, for devices with extremely limited processing and memory, this is necessary to even run on larger images in the first place (having been tested up to 8k real and 32k simulated pictures). 

This application is most useful for large zones that need to be monitored occasionally, such as retail shelves at a distance or drone images taken from a height.


# Requirements

```
opencv-python
torch
numpy
```

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
