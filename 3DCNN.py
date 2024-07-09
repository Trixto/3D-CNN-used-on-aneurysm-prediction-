import numpy as np
import os
import math
import datetime
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import Sequence
# Updated mixed precision import
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

class DataGenerator(Sequence):
    def __init__(self, data_paths, labels, batch_size=16, num_of_classes=2, resolution=128, minV=None, maxV=None):
        self.data_paths = data_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        self.resolution = resolution
        self.minV = minV
        self.maxV = maxV

    def __len__(self):
        return math.ceil(len(self.data_paths) / self.batch_size)

    def __getitem__(self, idx):
        batch_paths = self.data_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_data = []
        for path in batch_paths:
            [X, Y, Z] = load_obj_file(path)
            [Xn, Yn, Zn] = normalizeVertexes(X, Y, Z, self.minV, self.maxV, self.resolution)
            cube = modelToCube(Xn, Yn, Zn, self.resolution)
            batch_data.append(cube)
        
        # Convert batch_data to np.float32 before expanding dimensions
        batch_data = np.array(batch_data).astype(np.float32)
        batch_data = np.expand_dims(batch_data, axis=4)
        
        # Ensure labels are also appropriately typed. This might not need conversion, depending on your setup.
        batch_labels = np.array(batch_labels).astype(np.float32)
        
        return batch_data, batch_labels

# Define your model architecture here
def build_model(input_shape, num_of_classes):
    model = models.Sequential([
        layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        layers.Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer='he_uniform'),
        layers.MaxPooling3D((2, 2, 2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(num_of_classes, activation='softmax', dtype='float32')  # Ensure the output layer uses float32
    ])
    return model

# Placeholder functions for loading and preprocessing your data
def load_obj_file(filename):
    X = []
    Y = []
    Z = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            s_line = line.split()
            if len(s_line) > 0:
                if s_line[0] == 'v':
                    X.append(float(s_line[1]))
                    Y.append(float(s_line[2]))
                    Z.append(float(s_line[3]))
    return X, Y, Z

def normalizeVertexes(X, Y, Z, minV, maxV, resolution):
    # prazdne listy pre navratove hodnoty
    Xret = []
    Yret = []
    Zret = []
	# pre kazdy vertex ...
    for (x, y, z) in zip(X, Y, Z):	
        # normalizujeme
        Xnorm = (resolution-1)*((x-minV)/(maxV-minV))
        Ynorm = (resolution-1)*((y-minV)/(maxV-minV))
        Znorm = (resolution-1)*((z-minV)/(maxV-minV))
        Xret.append(Xnorm)
        Yret.append(Ynorm)
        Zret.append(Znorm)
	# vratime
    return Xret, Yret, Zret
def modelToCube(Xn, Yn, Zn, resolution):
    # nulova kocka podla rozlisenia
	kocka = np.zeros((resolution, resolution, resolution),dtype='object')
	# pre kazdy vertex ...
	for (x, y, z) in zip(Xn, Yn, Zn):	
		# tam kde v kocke je vertex dame 1
		kocka[int(math.floor(x)), int(math.floor(y)), int(math.floor(z))] = 1
	# vratime
	return kocka

# Data paths and labels preparation
data_folder = 'C:\\Users\\HeavyHorse1\\Documents\\Skrvan\\LR\\IntrA\\generated\\'
data_paths = []  # Populate this with paths to your .obj files
labels = []  # Populate this with your labels
resolution = 128
label = 0
classes_list = os.listdir(data_folder)
# pocet tried je pocet podpriecinkov
num_of_classes = len(classes_list)
# pocitadlo vsetkych suborov
i = 0
# list vsetkych hodnot z ktorych sa bude pocitat extrem
hodnoty = []
# pre vsetky triedy ...
for c in classes_list:
	# zobraz meno triedy
    print(c)
	# zoznam suborov v triede
    files_list = os.listdir(os.path.join(data_folder, c))
    for f in files_list:
		# zobraz cestu suboru na spracovanie
        data_paths.append(os.path.join(os.path.join(data_folder, c), f))

def calculate_min_max(data_paths):
    all_vertexes = []
    for path in data_paths:
        [X, Y, Z] = load_obj_file(path)
        all_vertexes.extend(X)
        all_vertexes.extend(Y)
        all_vertexes.extend(Z)
    return min(all_vertexes), max(all_vertexes)


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Assuming data_paths is populated with the paths to your .obj files
Minimum, Maximum = calculate_min_max(data_paths)

for c in classes_list:
	# zoznam suborov v triede
    files_list = os.listdir(os.path.join(data_folder, c))
	# pre kazdy subor ...
    for f in files_list:
        # pridanie labelu do zoznamu
        labelVector = np.zeros(shape=(num_of_classes))
        labelVector[label]=1;
        labels.append(labelVector)
		
	# podpriecinok(trieda) je spracovany, label navysime o 1
    label += 1



# Splitting dataset into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(data_paths, labels, test_size=0.30, random_state=42)

# Instantiate the data generators
train_generator = DataGenerator(X_train, y_train, batch_size=4, num_of_classes=num_of_classes, resolution=resolution, minV=Minimum, maxV=Maximum)
validation_generator = DataGenerator(X_test, y_test, batch_size=4, num_of_classes=num_of_classes, resolution=resolution, minV=Minimum, maxV=Maximum)

# Model Compilation and Training
model = build_model((resolution, resolution, resolution, 1), num_of_classes)
optimizer = optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Training callbacks
checkpoint_filepath = 'model_checkpoint2.hdf5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=7)

# Train the model
history=model.fit(train_generator, epochs=20, validation_data=validation_generator, callbacks=[model_checkpoint_callback, early_stopping])
plot_training_history(history)

# Generate predictions for the test set
test_generator = DataGenerator(X_test, y_test, batch_size=4, num_of_classes=num_of_classes, resolution=resolution, minV=Minimum, maxV=Maximum)
Y_pred = model.predict(test_generator)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(y_test, axis=1)  # Assuming y_test are one-hot encoded true labels

# Compute the confusion matrix
cm = confusion_matrix(Y_true, Y_pred_classes)

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Actual label")
plt.xlabel("Predicted label")
plt.show()
