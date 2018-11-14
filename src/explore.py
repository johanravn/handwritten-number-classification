import numpy as np
import cv2
import hashlib
from dataset import load_annotations, BatchGenerator
from dataset import convert_to_categorical, create_cls_mapping
from sklearn.utils import shuffle

def load_dataset():

    annotations, classes = load_annotations()

    targets = np.array(annotations)[:, 1]
    print(targets)
    cls_mapping, cls_list = create_cls_mapping(targets)

    annotations = shuffle(annotations, random_state=52)
    # train, valid = train_test_split(annotations,
    #                                 test_size=0.10,
    #                                 random_state=52)
    return annotations

def check_duplicated_images(annotations):
    hashes = []
    dups = {}
    i = 0
    for file_path, y in annotations:
        i += 1
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(annotations)))
        hash = hashlib.md5(cv2.imread(file_path)).hexdigest()
        hashes.append(hash)
    print(len(hashes))
    print(len(set(hashes)))

def check_duplicates_annotations(annotations):


    # Setup train data
    y = np.array(annotations)[:, 1]
    x = np.array(annotations)[:, 0]

    print("check duplicated files")
    print(len(x))
    print(len(set(x)))
    dups = {}
    for elem in x:
        if elem not in dups:
            dups[elem] = 1
        else:
            dups[elem] += 1
    for key, value in dups.items():
        # if value >= 2:
        if value > 2:
            print(key, value)
if __name__== "__main__":
    dataset = load_dataset()
    check_duplicates_annotations(dataset)
    check_duplicated_images(dataset)
