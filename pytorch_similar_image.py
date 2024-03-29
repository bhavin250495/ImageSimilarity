import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import math
import csv

import Utils



# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')
model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

class Image_Similarity:

    def __init__(self,similarity,image_path):
        self.similarity = similarity
        self.image_path = image_path

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    #t_img = t_img.view(1, -1)

    print(t_img)
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding

def get_image_vector_dict(path):
    dict = {}
    for idx, dirname, files in os.walk(path):
        for file in files:

            img_path = path + file
            vector = get_vector(img_path)
            dict[img_path] = vector
    return dict

def find_similarity_percent(vector_1,vector_2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_sim = cos(vector_1.unsqueeze(0),
                  vector_2.unsqueeze(0))
    return cos_sim


def get_similar_images(image_vector, dict_of_path_and_vector,threshold):
    similar_image_path = []

    for key, val in dict_of_path_and_vector.items():
        similarity = find_similarity_percent(image_vector,val)

        if similarity > threshold:
            image = Image_Similarity(similarity,key)
            similar_image_path.append(image)


    return similar_image_path

def showImages(image_path_array):

    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    image_cnt = int(len(image_path_array))
    columns = 5
    rows = 1
    if image_cnt > 5:
        rows = math.ceil(image_cnt/5)

    for i in range(1, columns * rows + 1):
        try:

            print(image_path_array[i-1])
            fig.add_subplot(rows, columns, i).set_title("{0:.2f}".format(float(image_path_array[i-1].similarity)))
            img = mpimg.imread(image_path_array[i-1].image_path)
            plt.imshow(img)
        except:
            print('out of range')
    plt.show()

def get_max_similarity_items(similar_image_dict,count):
   list =  sorted(similar_image_dict, key=lambda image:image.similarity)
   if len(list) > count:
       return list[-count:]


