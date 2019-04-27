import torch
import numpy as np
from PIL import Image
from torchvision import transforms


def load_image(img_path, img_size=None):
    '''
        Resize the input image so we can make content image and style image have same size, 
        change image into tensor and normalize it
    '''
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        # convert the (H x W x C) PIL image in the range(0, 255) into (C x H x W) tensor in the range(0.0, 1.0) 
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # this is from ImageNet dataset
                        ])   

    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    '''
        tensor (batch, channel, height, width)
        numpy.array (height, width, channel)
        to transform tensor to numpy, tensor.transpose(1,2,0) 
    '''
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image


def get_features(image, model, layers=None):
    '''
        return a dictionary consists of each layer's name and it's feature maps
    '''
    if layers is None:
        layers = {'0': 'conv1_1',   # default style layer
                  '5': 'conv2_1',   # default style layer
                  '10': 'conv3_1',  # default style layer
                  '19': 'conv4_1',  # default style layer
                  '21': 'conv4_2',  # default content layer
                  '28': 'conv5_1'}  # default style layer
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)    #  layer(x) is the feature map through the layer when the input is x
        if name in layers:
            features[layers[name]] = x
    
    return features


def get_grim_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(b * c, h * w)
    gram_matrix = torch.mm(tensor, tensor.t())
    return gram_matrix