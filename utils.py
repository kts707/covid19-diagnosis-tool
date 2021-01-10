# import dependencies
import torch


# utility functions
def ignore_noncovid(path):
    if path.find("non_covid") == -1:
        return True
    else:
        return False

def ignore_nii(path):
    if path.find("covid_mask_png") == -1 or path.find("outputFile.csv") != -1:
        return False
    else:
        return True

def iou_pytorch(outputs, labels):
    
    SMOOTH = 1e-6
    # print('raw',outputs.shape)
    outputs = torch.argmax(outputs, 1)
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    # print('output=',outputs)
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    # print('intersection=',intersection)
    # print('union=',union)
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # smooth our devision to avoid 0/0
    # print('iou=',iou)
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  
    
    return thresholded.mean() 


def convert_to_binary(masks, thres=0.5):
    binary_masks = ((masks[:, 0, :, :] ==  128) & (masks[:, 1, :, :] == 0) & (masks[:, 2, :, :] == 0)) + 0.
    return binary_masks.long()


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self