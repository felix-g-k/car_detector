import os
os.system("python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")

import gradio as gr
import torch.nn
import torch
from torchvision import datasets, models, transforms
from PIL import Image
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import cv2
import torch.nn.functional as F
import json
from torchvision.transforms.functional import crop, to_pil_image

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# App description and title
title = "Car detection"
description = "Image segmentation and detection of automobile instances"

def segment_img(img):
    
    # Convert the Gradio image (PIL image) to a NumPy array
    img_np = np.array(img)

    ## -- PART 2 -- ##

    # building the configuration for the image segmentation model (model used: mask_rcnn_R_50_FPN_3x)
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu' 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

    # initialize predictor
    predictor = DefaultPredictor(cfg)

    # Use the Detectron2 predictor to get the segmentation outputs
    outputs = predictor(img_np)

    # Get the instances from the outputs
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.cpu().tolist()

    
    ## -- PART 3 -- ##

    # If there are no car instances detected, return a message
    if 2 not in pred_classes:
        return ['No car detected in image', None, None, None]


    ## -- PART 4 -- ##
        
    # salient instance = instance with largest number of pixels --> largest area
    max_area = 0
    # by default, the salient index is the first instance
    class_indexes = np.where(np.array(pred_classes) == 2)
    salient_instance_idx =  np.min(class_indexes)

    areas = instances.pred_boxes.area()

    # loop through detected number of instances in PART 3 and get the area and the index of the instance with the largest area
    for i in range(areas.size()[0]):
        if pred_classes[i] == 2:  # Class 2 corresponds to "car" in COCO dataset
            area = areas[i]
            if area > max_area:
                max_area = area
                salient_instance_idx = i

    if salient_instance_idx is None:
        return ["No salient car visible.", None, None, None]

    # This part is unneccessary, but I wanted to show all detected instances as intermediate step
    metadata = MetadataCatalog.get(predictor.cfg.DATASETS.TRAIN[0])
    v = Visualizer(img_np[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(instances)
    segmented_image = out.get_image()[:, :, ::-1]

    
    ## -- PART 5 -- #
    
    # use the salient instance index to create a mask of the object and return the image
    sal_mask_tensor = outputs["instances"].pred_masks[salient_instance_idx].cpu()
    sal_mask = np.array(sal_mask_tensor) # convert to np array

    # Create a 4-channel image with the mask in the alpha channel, use mask to control transparency of car pixels and background pixels
    img_4ch = np.zeros((sal_mask.shape[0], sal_mask.shape[1], 4), dtype=np.uint8)
    img_4ch[:, :, 0] = img_np[:, :, 0]  # Red channel
    img_4ch[:, :, 1] = img_np[:, :, 1]  # Green channel
    img_4ch[:, :, 2] = img_np[:, :, 2]  # Blue channel
    img_4ch[:, :, 3] = np.where(sal_mask, 255, 0)  # Alpha channel

    # cropping image
    bbox = instances.pred_boxes[salient_instance_idx].tensor[0].tolist()
    img_rgb = img_4ch[:, :, :3]
    cropped_img_4ch = crop(img_rgb, bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])
    
    ## -- PART 6 -- ## 

    # specify model architecture adapted to the binary classification task
    viewpoint_model = models.resnet18(pretrained=True)
    viewpoint_model.fc = torch.nn.Linear(viewpoint_model.fc.in_features, 2)

    # load state dict of trained model
    #viewpoint_model.load_state_dict(torch.load('viewpoint_model_230723_model_sample.pt', map_location='cpu'))
    viewpoint_model.load_state_dict(torch.load('viewpoints_50000_20230725_bal_model.pt', map_location='cpu'))
    viewpoint_model.to(device)
    viewpoint_model.eval()

    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    # predict viewpoint of image
    mdl_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Resize(224),
                           #transforms.ToTensor(),
                           transforms.Normalize(mean=imagenet_mean,
                                                std=imagenet_std)
                       ])
    
    transf_img = mdl_transforms(cropped_img_4ch)
    
    with torch.no_grad():
        transf_img = transf_img.to(device)
        transf_img = transf_img.unsqueeze(0)
        # Get the prediction probabilities, the highest is the viewpoint
        pred_label = viewpoint_model(transf_img)
        viewpoint_pred = torch.argmax(pred_label, 1)


    # stop evaluation and print message that car needs to be frontal view
    if viewpoint_pred == 0:
        return ["Car detected!", "Number of detected instances: {}".format(areas.size()[0]), segmented_image, img_4ch, 
                "No frontal view detected. Model is trained on full frontal views!"]


    ## -- PART 7 -- ##

    # - load models from previous assignment - # 

    # mappings for turning predictions into bodytype and year categories
    year_range_mapping = {
        0: "2000-2003",
        1: "2006-2008",
        2: "2009-2011",
        3: "2012-2014",
        4: "2015-2018"
    }

    btype_mapping = {
        0: "Hatchback",
        1: "SUV",
        2: "MPV", 
        3: "Convertible",
        4: "Saloon"
    }

    # modernity score
    mscore_model = models.resnet18(pretrained=True)
    mscore_model.fc = torch.nn.Linear(mscore_model.fc.in_features, 5)
    mscore_model.load_state_dict(torch.load('modernity_score_ResNet18_v2.pt', map_location='cpu'))
    mscore_model.to(device)
    mscore_model.eval()

    # bodytype
    btype_model = models.resnet18(pretrained=True)
    btype_model.fc = torch.nn.Linear(btype_model.fc.in_features, 5)
    btype_model.load_state_dict(torch.load('bodytypes_ResNet18.pt', map_location='cpu'))
    btype_model.to(device)
    btype_model.eval()

    # - modernity score inference - # 

    with torch.no_grad():
        
        mscore_pred = mscore_model(transf_img)
        mscore = torch.sum(F.softmax(mscore_pred, dim=1) * torch.tensor([0,1,2,3,4], device=device), dim=1)
    
    year_pred = year_range_mapping[round(mscore.item())]

    # - typicality score inference - #
        
    # load group morphs dictionary from third assignment
    with open("group_morphs.json", 'r') as fp:
        morphs_d = json.load(fp)

    morphs = {key: torch.tensor(morphs_d[key]).reshape(-1, 1, 1) for key in morphs_d.keys()}

    # get activation via hook as in last assignment
    def get_activations(name):
      # the hook signature
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook
        
    activation = {}
    hook = btype_model.avgpool.register_forward_hook(get_activations('avgpool'))

    # predict bodytype
    with torch.no_grad():
        btype_lbls = btype_model(transf_img)
        btype_pred = btype_mapping[torch.argmax(btype_lbls, 1).item()]
        img_activation = activation['avgpool']

    hook.remove()
    
    group = btype_pred + " - " + year_pred

    # compute typicality score as cosine similarity between input activation and group morph
    typ_scr = torch.nn.CosineSimilarity(dim=0)(activation['avgpool'].squeeze(), morphs[group].squeeze()).item()
    
    # Return the segmented image and the text output
    return ["Car detected!", "Number of detected instances: {}".format(areas.size()[0]), segmented_image, img_4ch, 
            "Predicted category: {} (Typicality score: {})".format(group, round(typ_scr, 2))]

examples = [['frank_ocean_nostalgia_ULTRA.jpg'], ['GKMC_cover.jpg'], ['multiple_cars_img.jpg'], ['lightning_mcqueen.jpeg'], ['new_york_traffic.jpeg'],
           ['barbiemobile.jpg'], ['BMW_frontal.jpg'], ['hasbullah_g_class.jpg'], ['hasbullah_VW.jpeg'], ['hasbullah_bugatti.jpeg']]

    
interface = gr.Interface(segment_img, inputs='image', outputs=["label", "label", "image", "image", "label"], examples=examples, title=title, description=description, cache_examples=False)
interface.launch()