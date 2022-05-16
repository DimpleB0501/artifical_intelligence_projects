import torch
import torchvision
import numpy as np
import cv2
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
import matplotlib.pyplot as plt
import utils

# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# initialize the model
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                               num_keypoints=17)
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model.to(device).eval()

image = Image.open('/home/dimple/Desktop/pose_estimation/input/task1_in.png').convert('RGB')
# NumPy copy of the image for OpenCV functions
orig_numpy = np.array(image, dtype=np.float32)
# convert the NumPy image to OpenCV BGR format
orig_numpy = cv2.cvtColor(orig_numpy, cv2.COLOR_RGB2BGR) / 255.
# transform the image
image = transform(image)
# add a batch dimension
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)
#print (outputs)
output_image = utils.draw_keypoints(outputs, orig_numpy)
# visualize the image
output_image=cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
plt.rcParams['figure.figsize'] = [20, 10]
plt.imshow(output_image)
cv2.imwrite('/home/dimple/Desktop/pose_estimation/output/output_image.jpg', output_image*255.)
plt.show()
