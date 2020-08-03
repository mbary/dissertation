

import os
import copy
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.cm as mpl_color_map



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer,device):
        self.model = model.to(device)
        self.target_layer = target_layer


    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        # print("x type", type(x))
        # print(x)

        # x = torch.unsqueeze(x,0).cuda()
        # print("x shape after unsqueeze", x.shape)
        for module_pos, module in self.model._modules.items():
            # print(module_pos)
            x = module(x)  # Forward
            # print(f"x shape after {module_pos}", x.shape)
            if module_pos == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
                return conv_output, x

    def forward_pass(self, x):

        # Forward pass on the convolutions
        # print("X shape before forward pass", x.shape)
        conv_output, x = self.forward_pass_on_convolutions(x)

        # Forward pass on the classifier
        # print("X shape in forward pass", x.shape)

        x = self.model.avgpool(x)
        # Redefine the FC to match the
        #conv layer and num of classes
        fc_in_feaures = x.shape[1]
        self.model.fc = nn.Linear(fc_in_feaures,65).cuda()

        x=x.view(x.size(0),-1)
        # print("x shape before fc",x.shape)
        x = self.model.fc(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer, device):
        self.model = model.to(device)
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer, device)

    def apply_colormap_on_image(self, filename, activation, input_image ,colormap_name="gnuplot2"):

        # print("original image type", type(filename))

        map_size = input_image.shape[2:]
        org_im = Image.open(filename).convert('RGB')
        org_im = org_im.resize(map_size)

        """
            Apply heatmap on image
        Args:
            org_img (PIL img): Original image
            activation_map (numpy arr): Activation map (grayscale) 0-255
            colormap_name (str): Name of the colormap
        """
        # Get colormap
        color_map = mpl_color_map.get_cmap(colormap_name)
        no_trans_heatmap = color_map(activation)
        # Change alpha channel in colormap to make sure original image is displayed
        heatmap = copy.copy(no_trans_heatmap)
        heatmap[:, :, 3] = 0.65
        heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
        no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

        # Apply heatmap on iamge
        heatmap_on_image = Image.new("RGBA",map_size)
        heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
        # print("shape of heatmap_on_image", heatmap_on_image.size)
        # print("shape of heatmap", heatmap.size)

        heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
        return no_trans_heatmap, heatmap_on_image

    def generate_cam(self, input_image, filename ,target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().clone().numpy())
        # Get convolution outputs
        target = conv_output[0]
        # print("target",target.shape)
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum

        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            input_size = input_image.shape[2:]
            # print("inputsize",input_size)
            saliency_map = F.interpolate(saliency_map, size=(input_size[0],input_size[0]), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
#             print("img shape", input_image.shape)
#             print("norm_saliency_map shape", norm_saliency_map.shape)
#             print("target class", target_class)
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.detach().cpu().numpy() * target[i, :, :].data.detach().cpu().clone().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255


        no_trans_heatmap, heatmap_on_image = self.apply_colormap_on_image(filename, cam, input_image)

        return no_trans_heatmap, heatmap_on_image