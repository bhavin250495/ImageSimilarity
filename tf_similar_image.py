
import numpy as np
import os
import ssl
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA


ssl._create_default_https_context = ssl._create_unverified_context

PATH = 'Dset/'


import Utils as util

base_model = util.named_model('MobileNet')


class Image_Similarity:

    def __init__(self,similarity,image_path,vector):
        self.similarity = similarity
        self.image_path = image_path
        self.vector = vector


def get_feature_vector_list():
    files = os.listdir(PATH)
    vector_array = []
    file_name = []
    for file in files:
        vector = util.get_feature_vector(PATH+file,base_model)
        vector_array.append(vector)
        file_name.append(file)
    return vector_array,file_name

def get_pca_of_image(image_path,base_model,pca_model):
    vector = util.get_feature_vector(image_path, base_model)
    pca_result = pca_model.transform(vector.reshape(1, -1))
    return pca_result.reshape(-1)

def find_similarity_percent(vector_1,vector_2):

    cos = cosine_similarity(np.array([vector_1,vector_2]))

    return cos[0][1]


def get_similar_images(image_vector, dict_of_path_and_vector,threshold):
    similar_image_path = []

    for key, val in dict_of_path_and_vector.items():
        similarity = find_similarity_percent(image_vector,val)


        if similarity > threshold:
            image = Image_Similarity(similarity,key,val)
            similar_image_path.append(image)


    return similar_image_path


def pca_to_dict(files,pca_result):
    dict = {}
    for idx,file in enumerate(files):
        dict[file] = pca_result[idx]
    return dict



