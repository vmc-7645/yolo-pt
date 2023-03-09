"""
LICENSE AGREEMENT
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install, modify, copy or use the software in any part, partial or whole.
Do not remove this license notice.
"""

## IMPORTS

# Image processing
import cv2
import torch

# Data management
import numpy as np

# Get adjacent zones
def get_adj_zone_locations(pos, array_dim=(3,2), addself=True):
    """
    Gets the adjacent locations of a given location in a 2d array as a 1d array

    Parameters
    ----------
    pos: Position of tile in 1d array
    array_dim: Dimensions of 1d array in 2ds
    addself: whether or not to include self position as adjacent to self position

    Return
    ----------
    Positions of element in 1d array of location 2d array adjacent to position of pos in 1d array.
    """
    x_pos = (pos%array_dim[0])
    y_pos = (pos%array_dim[1])
    zoned_arr = []
    counter = 0
    for i in range(0, array_dim[1]):
        y = []
        for j in range(0, array_dim[0]):
            y.append(counter)
            counter+=1
        zoned_arr.append(y)
    return_arr=[]
    for r in [-1, 0, 1]:
        for c in [-1, 0, 1]:
            if r == c == 0:
                continue
            if 0 <= x_pos+r < array_dim[0] and 0 <= y_pos+c < array_dim[1]:
                return_arr.append(zoned_arr[y_pos+c][x_pos+r])
    if addself:
        return_arr.append(pos)
    return return_arr

# Normalizes image for processing
def normalize(
        img, 
        img_mean=(78.4263377603, 87.7689143744, 114.895847746), 
        img_scale=1/256
        ):
    """
    Outputs a normalized version of the input image 

    Parameters
    ----------
    img: image to be normalized
    img_mean: mean of image to be normalized to
    img_scale: scale by which to normalize by

    Return
    ----------
    Normalized image
    """
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Author: Martin Thoma
    Source: https://stackoverflow.com/a/42874377/13171500

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_iou_arr(bb1, bb2):
    b1 = {'x1':bb1[0], 'x2':bb1[2], 'y1':bb1[1], 'y2':bb1[3]}
    b2 = {'x1':bb2[0], 'x2':bb2[2], 'y1':bb2[1], 'y2':bb2[3]}
    return get_iou(b1,b2)

def runpt(  image_data,
            sensitivity=0.3,
            overlap=0.5,
            modelloc='weights/retail/best.pt', 
            model_shape=(640,480),
            MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746),
            img_save=False,
            img_show=False,
            debug=True,
            upscale_img_mult=1
            ):
    """
    Runs Retail detection pytorch model on an image.

    Parameters
    ----------
    image_data: Either file location or image loaded with cv2.imread
    sensitivity: How sensitive do we want accuracy of the ai model to output
    overlap: How much overlap to remove
    modelloc: Model location, can be used with other .pt models (even those that aren't necessarily for retail identification)
    model_shape: (640,480)
    MODEL_MEAN_VALUES: straightforward title
    img_save: Not implemented, saves image to device
    img_show: Show image on completion
    debug: Whether to show debug or not
    upscale_img_mult: Can sometimes allow for higher accuracy in predictions, seems to depend on a per case basis however

    Return
    ----------
    {
        "detections": Number of retail items found,
        "data": Retail items,
        "image": Image of retail items
    }

    """
    
    # Model Shape
    model_x, model_y = model_shape
    mx_half, my_half = (int(model_x/2), int(model_y/2))

    # Load model
    model = torch.load(modelloc, map_location=torch.device("cpu")).get('model').float() # make sure model is loaded in correctly

    # Get image
    if type(image_data)==str:
        frame = cv2.imread(image_data)
    else:
        frame = image_data
    
    frame_height, frame_width, ch = frame.shape
    
    if upscale_img_mult != 1:
        frame = cv2.resize(frame, (int(frame_height*upscale_img_mult), int(frame_width*upscale_img_mult)))
        frame_height, frame_width, ch = frame.shape
    
    if debug:
        print("Frame size "+str(frame_width)+"x"+str(frame_height))

    # Ensure we are scanning whole screen by checking if there are "leftovers"
    weird_y = False
    weird_x = False
    if frame_height%my_half!=0:
        if debug:
            print("Irregular image height, adjusting parameters to better scan zones...")
        weird_y = True
    if frame_width%mx_half!=0:
        if debug:
            print("Irregular image width, adjusting parameters to better scan zones...")
        weird_x = True
    
    # Zones we are going to scan, we only want to do this math once
    zones_to_scan = [] # [[y,yend,x,xend]]

    for i in range(0,(int(frame_height/my_half)-1)):
        y = (my_half*i)
        y_end = (y+model_y)
        for j in range(0, (int(frame_width/mx_half)-1)):
            x = (mx_half*j)
            x_end = (x+model_x)
            zones_to_scan.append([y,y_end,x,x_end])
        if weird_x:
            zones_to_scan.append([y,y_end,frame_width-model_x,frame_width])
    if weird_y:
        for j in range(0, (int(frame_width/mx_half)-1)):
            x = (mx_half*j)
            x_end = (x+model_x)
            zones_to_scan.append([frame_height-model_y,frame_height,x,x_end])
        if weird_x:
            zones_to_scan.append([frame_height-model_y,frame_height,frame_width-model_x,frame_width])

    # Number of zones (total)
    num_zones = len(zones_to_scan)

    # Dimensions of array
    zone_array_dimensions = (int(frame_width/mx_half)-(0 if weird_x else 1), int(frame_height/my_half)-(0 if weird_y else 1))

    # Adjacent zones
    adj_zone_arr = [get_adj_zone_locations(pos, zone_array_dimensions) for pos in range(0,len(zones_to_scan))]
    
    if debug:
        print("Scaning zones...")
    
    temp_boxes = [[] for i in range(0,num_zones)]
    
    zone_loc = 0
    for zone in zones_to_scan:
        y, x = zone[0], zone[2]
        cimg = frame[y:zone[1], x:zone[3]]  # Crop image to where we are scaning
        if debug:
            print("Scaning zone: ("+str(x)+":"+str(zone[3])+", "+str(y)+":"+str(zone[1])+")")
        cimg = cv2.resize(cimg, model_shape, interpolation=cv2.INTER_CUBIC)
        cimg = normalize(cimg, MODEL_MEAN_VALUES, 1/256) # Model Mean Values are just a guess
        cimg = torch.from_numpy(cimg).permute(2,0,1).unsqueeze(0).float()
        output = model(cimg) # Run model on image
        arr = np.squeeze(output[0].detach().cpu().numpy())
        zoned_boxes = []
        for k in range(len(arr[0])):
            if arr[4][k] > sensitivity:
                add_to_zone_checked = True
                for checked_boxes in temp_boxes[zone_loc]:
                    check = get_iou_arr(checked_boxes,[int(arr[0][k])+x, int(arr[1][k])+y, int(arr[0][k]+arr[2][k])+x, int(arr[1][k]+arr[3][k])+y, arr[4][k]])
                    if check > overlap:
                        add_to_zone_checked = False
                        break
                if add_to_zone_checked:
                    temp_boxes[zone_loc].append([int(arr[0][k])+x, int(arr[1][k])+y, int(arr[0][k]+arr[2][k])+x, int(arr[1][k]+arr[3][k])+y, arr[4][k]])
        zone_loc+=1
    
    # Ensure that overlaping images aren't scanned twice
    if debug:
        print("Checking for overlap...")
    
    # temp_accepted_boxes = [[] for i in range(0,num_zones)]
    boxes = []

    # Get zones that are going to be checked in adjacent and same
    for zones_to_check in adj_zone_arr:
        if debug:
            print("Checking zone: "+str(zones_to_check[-1]))
        # Check each zone
        for zone_being_checked in zones_to_check:
            items_in_zone = temp_boxes[zone_being_checked]
            # Check items in zone if overlapping objects in other zones
            for i in items_in_zone:
                add_box = True
                for j in boxes: # TODO Make check against found objects in specific zones as opposed to all zones - minor efficiency upgrade at scale
                    check = get_iou_arr(i,j)
                    if check > overlap: # If overlaping
                        add_box = False
                        break
                if add_box:
                    boxes.append(i)
                    cv2.rectangle(frame, (i[0]-int(int(i[2]-i[0])/2), i[1]-int(int(i[3]-i[1])/2)), (i[2]-int(int(i[2]-i[0])/2), i[3]-int(int(i[3]-i[1])/2)), (255,0,0),2)
    
    if debug:
        print("Found "+str(len(boxes))+" retail item(s).")
    
    frame = cv2.resize(frame, (640*2,480*2), interpolation=cv2.INTER_CUBIC)
    
    if img_show:
        while True:
            # Displaying color frame with contour of motion of object
            cv2.imshow("Color Frame", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    return {"detections":len(boxes),"data":boxes,"image":frame}

if __name__=='__main__':

    runpt("test-images/Picture3.png", 
        sensitivity=0.6,
        overlap=0.3,
        img_show=True,
        upscale_img_mult=2
        )