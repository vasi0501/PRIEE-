from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers, Model
from sklearn.ensemble import RandomForestClassifier

from app import classifier

model = Sequential()

model.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Convolution2D(32, 3,  3, activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Convolution2D(64, 3,  3, activation = 'relu'))
model.add(MaxPooling2D(pool_size =(2,2)))

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.compile(
              optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Data/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'Data/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

feature_extractor = Model(inputs=model.input, outputs=model.get_layer('some_layer_name').output)
features = feature_extractor.predict(train_datagen)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

rf_classifier.fit(features)

predictions = rf_classifier.predict(test_set)


import h5py
classifier.save('Trained_Model.h5')

print(model.history.keys())
import matplotlib.pyplot as plt

# summarize history for accuracy
plt.plot(model.history['acc'])
plt.plot(model.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()








