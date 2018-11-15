import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
from dataset import load_annotations, BatchGenerator
from dataset import convert_to_categorical, create_cls_mapping
import keras.backend as K
from keras.applications import Xception, MobileNet, ResNet50, VGG16
from keras.utils import multi_gpu_model
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def test():

    # Preprocess
    annotations, classes = load_annotations()

    targets = np.array(annotations)[:, 1]
    print(targets)
    cls_mapping, cls_list = create_cls_mapping(targets)

    annotations = shuffle(annotations, random_state=52)
    train, valid = train_test_split(annotations,
                                    test_size=0.10,
                                    random_state=52)

    valid = valid[:1000]
    # Setup validation data
    y_valid = np.array(valid)[:, 1]
    x_valid = np.array(valid)[:, 0]
    Y_valid = convert_to_categorical(y_valid, len(classes), cls_mapping)
    valid_generator = BatchGenerator(x_valid, Y_valid, 16, (500, 100))
    print("length of train valid_generator", len(valid_generator))

    # Set up model
    base_model = ResNet50(
                          include_top=False,
                          input_shape=(100, 500, 3),
                          classes=len(classes),
                         )
    x = base_model.output
    x = Flatten()(x)
    x = Dense(len(classes), activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights("../weights/100.h5")

    print("tart predicting on validation data")
    pred = model.predict_generator(valid_generator)
    np.save("pred", pred)
    np.save("y_valid", Y_valid)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    print(classification_report(np.array(Y_valid), pred, target_names=cls_list, digits=3))

if __name__ == "__main__":
    test()
