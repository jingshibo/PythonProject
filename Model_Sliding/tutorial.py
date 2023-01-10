# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.python import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.minivggnet import MiniVGGNet
import matplotlib.pyplot as plt
import numpy as np
##
def csv_image_generator(inputPath, bs, lb, mode="train", aug=None):
    # open the CSV file for reading
    f = open(inputPath, "r")

    # loop indefinitely
    while True:
        # initialize our batches of images and labels
        images = []
        labels = []

        # keep looping until we reach our batch size
        while len(images) < bs:
            # attempt to read the next line of the CSV file
            line = f.readline()

            # check to see if the line is empty, indicating we have
            # reached the end of the file
            if line == "":
                # reset the file pointer to the beginning of the file
                # and re-read the line
                f.seek(0)
                line = f.readline()

                # if we are evaluating we should now break from our
                # loop to ensure we don't continue to fill up the
                # batch from samples at the beginning of the file
                if mode == "eval":
                    break

            # extract the label and construct the image
            line = line.strip().split(",")
            label = line[0]
            image = np.array([int(x) for x in line[1:]], dtype="float32")
            image = image.reshape((64, 64, 3))

            # update our corresponding batches lists
            images.append(image)
            labels.append(label)

        # one-hot encode the labels
        labels = lb.transform(np.array(labels))
        images = np.array(images)
        # if the data augmentation object is not None, apply it
        if aug is not None:
            (images, labels) = next(aug.flow(images,
                labels, batch_size=bs))

        # yield the batch to the calling function
        yield (images, labels)
##
# initialize the paths to our training and testing CSV files
TRAIN_CSV = "flowers17_training.csv"
TEST_CSV = "flowers17_testing.csv"

# initialize the number of epochs to train for and batch size
NUM_EPOCHS = 75
BS = 32

# initialize the total number of training and testing image
NUM_TRAIN_IMAGES = 0
NUM_TEST_IMAGES = 0

# open the training CSV file, then initialize the unique set of class
# labels in the dataset along with the testing labels
f = open(TRAIN_CSV, "r")
labels = set()
testLabels = []

# loop over all rows of the CSV file
for line in f:
    # extract the class label, update the labels list, and increment
    # the total number of training images
    label = line.strip().split(",")[0]
    labels.add(label)
    NUM_TRAIN_IMAGES += 1

# close the training CSV file and open the testing CSV file
f.close()
f = open(TEST_CSV, "r")

# loop over the lines in the testing file
for line in f:
    # extract the class label, update the test labels list, and
    # increment the total number of testing images
    label = line.strip().split(",")[0]
    testLabels.append(label)
    NUM_TEST_IMAGES += 1

# close the testing CSV file
f.close()
##
# create the label binarizer for one-hot encoding labels, then encode
# the testing labels
lb = LabelBinarizer()
lb.fit(list(labels))
testLabels = lb.transform(testLabels)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")

# initialize both the training and testing image generators
trainGen = csv_image_generator(TRAIN_CSV, BS, lb,
	mode="train", aug=aug)
testGen = csv_image_generator(TEST_CSV, BS, lb,
	mode="train", aug=None)

# initialize our Keras model and compile it
model = MiniVGGNet.build(64, 64, 3, len(lb.classes_))
opt = optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2 / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training w/ generator...")
H = model.fit(
	x=trainGen,
	steps_per_epoch=NUM_TRAIN_IMAGES // BS,
	validation_data=testGen,
	validation_steps=NUM_TEST_IMAGES // BS,
	epochs=NUM_EPOCHS)

# re-initialize our testing data generator, this time for evaluating
testGen = csv_image_generator(TEST_CSV, BS, lb,
	mode="eval", aug=None)

# make predictions on the testing images, finding the index of the
# label with the corresponding largest predicted probability
predIdxs = model.predict(x=testGen, steps=(NUM_TEST_IMAGES // BS) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print("[INFO] evaluating network...")
print(classification_report(testLabels.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


##
class PrintNumber:
    def __init__(self, max):
        self.max = max
    def __iter__(self):
        self.num = 0
        return self
    def __next__(self):
        if(self.num >= self.max):
            raise StopIteration
        self.num += 1
        return self.num


##
import tensorflow as tf
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

