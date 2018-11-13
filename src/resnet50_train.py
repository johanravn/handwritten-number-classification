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

THRESHOLD = 0.05

# Metrics that weighs all classes the same
def f_macro_score(y_true, y_pred):
    #y_pred = K.round(y_pred)
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def train():

    # Preprocess
    annotations, classes = load_annotations()

    targets = np.array(annotations)[:, 1]
    print(targets)
    cls_mapping, cls_list = create_cls_mapping(targets)

    annotations = shuffle(annotations, random_state=52)
    train, valid = train_test_split(annotations,
                                    test_size=0.10,
                                    random_state=52)

    # Setup train data
    y_train = np.array(train)[:, 1]
    x_train = np.array(train)[:, 0]
    Y_train = convert_to_categorical(y_train, len(classes), cls_mapping)
    train_generator = BatchGenerator(x_train, Y_train, 24, (100, 500))
    print("length of train train_generator", len(train_generator))

    # Setup validation data
    y_valid = np.array(valid)[:, 1]
    x_valid = np.array(valid)[:, 0]
    Y_valid = convert_to_categorical(y_valid, len(classes), cls_mapping)
    valid_generator = BatchGenerator(x_valid, Y_valid, 24, (100, 500))
    print("length of train valid_generator", len(train_generator))

    # Set up model for training 
    with tf.device('/cpu:0'):
        base_model = ResNet50(
                        include_top=False,
                        #weights='imagenet',
                        input_shape=(500, 100, 3),
                        classes=len(classes),
                        )
        x = base_model.output
        x = Flatten()(x)
        x = Dense(len(classes), activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=x)
        # model.load_weights("../models/ResNet50_all_epoch_6.h5")
    parallel_model = multi_gpu_model(model, gpus=2)
    
    #parallel_model = model
    adam = Adam(lr=0.0001, amsgrad=True)
    #parallel_model.load_weights("weights/22.h5")

    parallel_model.compile(loss="categorical_crossentropy",
                           optimizer=adam,
                           metrics=['accuracy', f_macro_score])

    print("tart training")
    for i in range(1, 200):
        print(i)
        parallel_model.fit_generator(train_generator,
                                     steps_per_epoch=1000,
                                     epochs=1,
                                     verbose=1,
                                     validation_data=valid_generator,
                                     # validation_steps=1586,
                                     max_queue_size=200,#
                                     workers=6,
                                     use_multiprocessing=True)
        pred = parallel_model.predict_generator(valid_generator)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        print(classification_report(np.array(Y_valid), pred, target_names=cls_list))

        #model.save_weights("weights/"+str(i)+".h5", overwrite=True)

if __name__ == "__main__":
    train()
