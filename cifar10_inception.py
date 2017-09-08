from keras.datasets import cifar10
from keras.applications import InceptionV3
from keras.layers import Dense,Activation,Flatten,Reshape
from keras.models import Sequential,Model
from keras.callbacks import TensorBoard,ModelCheckpoint
import cv2
import numpy as np

# Model name
model_name = "inception_cifar10"
min_dim = 150

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train_scaled = np.zeros((X_train.shape[0], min_dim, min_dim, 3))
for i, img in enumerate(X_train):
    X_train_scaled[i] = cv2.resize(img, dsize=(min_dim, min_dim), interpolation=cv2.INTER_CUBIC)

X_test_scaled = np.zeros((X_test.shape[0], min_dim, min_dim, 3))
for i, img in enumerate(X_test):
    X_test_scaled[i] = cv2.resize(img, dsize=(min_dim, min_dim), interpolation=cv2.INTER_CUBIC)

y_train = np.eye(10)[y_train].reshape([50000,10])
y_test = np.eye(10)[y_test].reshape([10000,10])

print(X_train_scaled.shape)
print(y_train.shape)

inception_model = InceptionV3(weights='imagenet',include_top=False,input_shape=(min_dim,min_dim,3))
print("Number of layers needs to be excluded from training: {}".format(len(inception_model.layers)))

# Finetuning model
topmodel = GlobalAveragePooling2D(name='avg_pool')(topmodel)
topmodel = Dense(10, activation='softmax', name='predictions')(topmodel)

model = Model(inputs=inception_model.input,outputs=topmodel)

print(model.summary())

# Callbacks
tensorboardclbk = TensorBoard(log_dir='./logs/' + model_name,
                              histogram_freq=5,
                              batch_size=256,
                              write_graph=False,
                              write_grads=True,
                              write_images=False)

filepath= model_name + "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

modelcheckclbk = ModelCheckpoint("./checkpoints/" + filepath + ".h5", monitor='val_acc', verbose=0, save_best_only=True)


model.compile(loss="mean_squared_error",optimizer="adam",metrics=["acc"])

model.fit(X_train_scaled,
          y_train,
          batch_size=256,
          epochs=200,
          validation_data=(X_test_scaled,y_test),
          callbacks=[tensorboardclbk,modelcheckclbk])

