# simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception
# last update by BZ, June 30, 2021

import io
from matplotlib import image
import torch
from PIL import Image
from torchvision import models, transforms
from resnet18 import resnet18
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

# input image
#LABELS_file = 'imagenet-simple-labels.json'
#image_file = 'test.jpg'

# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = resnet18(num_classes=5, grayscale=False)
    net.load_state_dict(torch.load(r'C:\Users\33602\Desktop\eecs545_project\resnet18_sia_best.pth'))
    finalconv_name = 'layer4' # this is the last conv layer of the network
elif model_id == 2:
    net = resnet18(num_classes=5, grayscale=False)
    net.load_state_dict(torch.load(r'C:\Users\33602\Desktop\eecs545_project\resnet18_best.pth'))
    finalconv_name = 'layer4' 
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'

net.eval()

# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

net._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-4].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (28, 28)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
mean=[.5], std=[.5]
)
preprocess = transforms.Compose([
   transforms.Resize((28,28)),
   transforms.ToTensor(),
   normalize
])

# load test image
for k in range(399):
    image_file = r'C:\Users\33602\Desktop\retina' + '\\' + str(k) + '.jpg'
    img_pil = Image.open(image_file)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    logit, _ = net(img_variable)

    # load the imagenet category list
    #with open(LABELS_file) as f:
    #    classes = json.load(f)
    classes = {0:'Healthy', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'PDR'}

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # output the prediction
    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # generate class activation mapping for the top1 prediction
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
    img = cv2.imread(image_file)
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(r'C:\Users\33602\Desktop\retina_cam_pure\CAM' + str(k) + '.jpg', result)

