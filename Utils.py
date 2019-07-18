
from keras import applications
from keras.preprocessing import image
import numpy as np
from keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg




def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')


def get_feature_vector(img_path,base_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    # Model will take 4 dimensional input
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = base_model.predict(img_data)
    feature_np = np.array(feature)
    return feature_np.flatten()


def get_max_similarity_items(similar_image_dict,count):
   list =  sorted(similar_image_dict, key=lambda image:image.similarity)
   if len(list) > count:
       return list[-count:]


def showImages(base_path,image_path_array):

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
            img = mpimg.imread(base_path + image_path_array[i-1].image_path)
            plt.imshow(img)
        except:
            print('out of range')
    plt.show()


def show_cluster_with_labels(file_names,principalComponents):
    types = file_names
    x_coords = principalComponents[:, 0]
    y_coords = principalComponents[:, 1]

    for i, type in enumerate(types):
        x = x_coords[i]
        y = y_coords[i]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x + 0.3, y + 0.3, type, fontsize=9)
    plt.show()