import os
import cv2
import glob
import numpy as np

from imgaug import augmenters as iaa
from keras.utils import Sequence
import random

random.seed(52)
sample_minimum = 5
# drop chance for empty class
# make a more reasonable split
drop_chance = 0.90


def create_cls_mapping(targets):
    cls_mapping = {}
    unique_targets = np.unique(targets)
    for i in range(0, len(unique_targets)):
        cls_mapping[unique_targets[i]] = i
    print(cls_mapping)
    return cls_mapping, unique_targets

def convert_to_categorical(targets, num_classes, cls_mapping):
    new_y = []
    print(num_classes)
    for target in targets:
        tmp = np.zeros(num_classes)
        # Look up correct id for class
        tmp[cls_mapping[str(target)]] = 1
        new_y.append(tmp)
    return new_y


def create_image(path, shape):
    #print(path)
    img = cv2.imread(path)
    #print(img.shape)
    try:
        # Cv2 uses flipped shape, should 
        img = cv2.resize(img, (shape[0], shape[1]))
    except cv2.error as e:
        #print(e)
        print(path)
        return None
    return img


def remove_low_class_count(annotations, freq_dict):
    new_anno = []
    num_classes = 0
    for file_path, target in annotations:
        if freq_dict[target] >= sample_minimum:
            new_anno.append([file_path, target])
    classes = []
    for cls in freq_dict:
        if freq_dict[cls] >= sample_minimum:
            classes.append(cls)
    return new_anno, classes

def check_if_image_exits(annotations):
    new_anno = []
    count = 0
    for file_path, target in annotations:
        if not os.path.isfile(file_path):
            count += 1
            continue
        new_anno.append([file_path, target])
    print("annotations removed - missing image file: ", count)
    return new_anno

def remove_duplicates(annotations):
    new_anno = []

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
    for anno in annotations:
        if dups[anno[0]] == 1:
            new_anno.append(anno)
    return new_anno


def get_class_frequency(array):
    array = np.array(array)
    print(array)
    y = np.bincount(array)
    ii = np.nonzero(y)[0]
    freq_table = zip(ii, y[ii])
    freq_dict = {}
    count = 0
    low_sample_classes = 0
    for i in freq_table:
        freq_dict[i[0]] = i[1]
        if i[1] < sample_minimum:
            low_sample_classes += 1
            continue
        count += 1
    print("low sample classes", low_sample_classes)
    print("total unique usable classes", count)
    return freq_dict

def extract_path_and_target(line):
    if line.find('label') != -1:
        return None
    if line == '\n':
        return None
    line = line.split(",")
    # skip empty lines
    file_path = line[0]
    target = line[1].strip('\n')
    if target == '':
        return None
        if random.uniform(0, 1) < drop_chance:
            return None
        target = 0
    # skip misformatted target strings
    try:
        target = int(target)
    except:
        print(target)
        return None
    return file_path, target

def convert_to_single_digits(annotations):
    new_anno = []
    count = 0
    for file_path, three_digit in annotations:
        three_digit = random.randint(100, 999)
        directories = file_path.split("/")
        new_file_path = "../input/single_digits/"
        new_file_path += directories[-1].split(".")[0] + "/"
        #print(new_file_path)
        # check that the directory exits
        if not os.path.isdir(new_file_path):
            count += 1
            continue
        # check that there are three images in the directory
        if len(glob.glob(new_file_path + "*")) != 3:
            count += 1
            continue
        for i in range(len(str(three_digit))):

            # Manually combine new path..
            path = new_file_path + str(i) + "_" + directories[-1]
            new_anno.append([path, int(str(three_digit)[i])])
    print("folders that don't exist or that does not have three images", count)
    return new_anno

def load_annotations():
    #labels = open("../input/" + dataset_name + ".csv")
    label_files = glob.glob("../input/labels/*")
    print(label_files)
    annotations = []
    all_targets = []

    for file in label_files:
        f = open(file)
        for i, line in enumerate(f):
            # skip csv header

            ret = extract_path_and_target(line)
            # skip faulty line
            if ret is None:
                continue
            file_path, target = ret
            # skip high frequency classes:
            #if target == 111 or target == 183 or target == 531 or target == 555 or target == 899:
            #    continue
            #print(file_path, target)
            directories = file_path.split('\\')
            # merge correct paths
            file_path = "../input/felt/"
            file_path += directories[len(directories) - 2]
            file_path += "/" + directories[len(directories) - 1]
            all_targets.append(int(target))
            annotations.append((file_path, target))

            #print(file_path)
    freq_dict = get_class_frequency(all_targets)

    useable_samples = [x for x in all_targets if x != 0]
    print("total number of useable samples", len(useable_samples))
    print("total number of samples", len(all_targets))
    annotations = remove_duplicates(annotations)
    annotations = convert_to_single_digits(annotations)
    annotations = check_if_image_exits(annotations)
    #annotations, classes = remove_low_class_count(annotations, freq_dict)
    print("final single digis set size:", len(annotations))
    classes = [str(x) for x in range(10)]
    return annotations, classes



# Generator for kears model. Yields batches.
class BatchGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size, shape):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shape = shape

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def horizontal_flip(self, x):
        if np.random.random() < 0.5:
            axis = 1
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
        return x

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = []
        Y = []
        seq = iaa.Sequential([
            iaa.OneOf([
                #iaa.Fliplr(0.5), # horizontal flips
                #iaa.Crop(percent=(0, 0.1)), # random crops
                # Small gaussian blur with random sigma between 0 and 0.5.
                # But we only blur about 50% of all images.
                iaa.Sometimes(0.5,
                    iaa.GaussianBlur(sigma=(0, 0.5))
                ),
                # Strengthen or weaken the contrast in each image.
                iaa.ContrastNormalization((0.75, 1.5)),
                # Add gaussian noise.
                # For 50% of all images, we sample the noise once per pixel.
                # For the other 50% of all images, we sample the noise per pixel AND
                # channel. This can change the color (not only brightness) of the
                # pixels.
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                # Make some images brighter and some darker.
                # In 20% of all cases, we sample the multiplier once per channel,
                # which can end up changing the color of the images.
                #iaa.Multiply((0.8, 1.2), per_channel=0.2),
                # Apply affine transformations to each image.
                # Scale/zoom them, translate/move them, rotate them and shear them.
                iaa.Affine(
                #    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-180, 180),
                    shear=(-8, 8)
                )
            ])], random_order=True)

        for img, y in zip(batch_x, batch_y):
            #print("img", img)
            img = create_image(img, self.shape)

            if img is None:
                continue

            img = seq.augment_image(img)
            X.append(img)
            Y.append(y)
        X = np.array(X).astype('float32')/255.0
        return np.array(X), np.array(Y)#, np.array(weights)

def load_validation_data(annotations, shape, cls_mapping):
    X = []
    y = []
    for file_path, target in (annotations):
        X.append(create_image(file_path, shape))
        y.append(target)
    y_set = convert_to_categorical(np.array(y), 10, cls_mapping)

    X = np.array(X).astype("float32")/255.0
    print(X.shape)
    return np.array(X), np.array(y_set)


if __name__ == "__main__":
    train, valid = load_annotations(0.33)
    x_set, y_set = generate_sets(train)
    create_skip_list(x_set)
