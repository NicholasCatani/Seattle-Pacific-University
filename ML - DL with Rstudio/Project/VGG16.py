##### Standard Libraries

import os

import keras.layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from keras.preprocessing.image import ImageDataGenerator

##### Deep Learning Libraries

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, Adamax
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras import regularizers

##### Convolutional Neural Network

from keras.applications.vgg16 import VGG16

##### Image Display

from PIL import Image



##########################################################
###################### FUNCTIONS #########################
##########################################################

## DataFrame
def create_df(dataset):
    image_paths, labels = [], []

    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:

            image = os.path.join(dirpath, filename)
            image_paths.append(image)
            if dirpath[-3:] == "all":
                labels.append("all")
            else:
                labels.append("hem")

    df = pd.DataFrame({"Image Path": image_paths,
                       "Label": labels})

    return df

train_dir = "C:\\Users\\Nicho\\Desktop\\C-NMC_Leukemia\\training_data"
df = create_df(train_dir)

train_df, remaining_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=31, stratify=df["Label"])
valid_df, test_df = train_test_split(remaining_df, train_size=0.5, shuffle=True, random_state=31, stratify=remaining_df["Label"])

print("Number of training samples: %d" % len(train_df.index))
print("Number of test samples: %d" % len(test_df.index))
print("Number of validation samples: %d" % len(valid_df.index))

## Model Performance
def show_history_plot(history):

    training_accuracy = history["accuracy"]
    epochs = range(1, len(training_accuracy) + 1)

    # Creating subplots for accuracy and loss
    plt.figure(figsize=(15, 5))

    # Plotting training and validation accuracy
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.plot(epochs, history["accuracy"], "b", label="Training accuracy", marker="o")
    plt.plot(epochs, history["val_accuracy"], "c", label="Validation Accuracy", marker="o")
    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)

    # Plotting training and validation loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(epochs, history["loss"], "b", label="Training loss", marker="o")
    plt.plot(epochs, history["val_loss"], "c", label="Validation loss", marker="o")
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True)

    # Improve layout and display the plot
    plt.tight_layout()
    plt.show()

## Confusion Matrix
def show_conf_matrix(model):
    test_gen.reset()  # Reset the generator to be sure it's at the start of the dataset
    y_pred = model.predict(test_gen, steps=test_gen.n // test_gen.batch_size + 1, verbose=0)

    label_dict = test_gen.class_indices
    classes = list(label_dict.keys())

    # Convert predictions to labels
    pred_labels = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes

    # Generate the confusion matrix
    confusion_matrix = metrics.confusion_matrix(y_true, pred_labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

    # Plot the confusion matrix
    cmap = plt.cm.Blues
    cm_display.plot(cmap=cmap, colorbar=False)

    plt.title('Confusion Matrix', fontsize=16)
    plt.figure(figsize=(7, 7))
    plt.show()

## Evaluation Matrix
def evaluation_matrix(model):
    test_steps = len(test_df) // batch_size
    train_score = model.evaluate(train_gen, steps=test_steps, verbose=0)
    valid_score = model.evaluate(valid_gen, steps=test_steps, verbose=0)
    test_score = model.evaluate(test_gen, steps=test_steps, verbose=0)

    header="{:<12} {:<10} {:<10}".format("", "Loss", "Accuracy")
    separator = "-" * len(header)
    train_row = "{:<12} {:<10.5f} {:<10.5f}".format("Train", train_score[0], train_score[1])  # Formats the float to five decimal places
    valid_row = "{:<12} {:<10.5f} {:<10.5f}".format("Validation", valid_score[0], valid_score[1])
    test_row = "{:<12} {:<10.5f} {:<10.5f}".format("Test", test_score[0], test_score[1])

    table = "\n".join([header, separator, train_row, valid_row, test_row])
    print(table)

## Show Images
hem_img = train_df[train_df["Label"] == "hem"].sample(6)
all_img = train_df[train_df["Label"] == "all"].sample(6)
sampled_df = pd.concat([hem_img, all_img])

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 6))

for i, row in enumerate(sampled_df.iterrows()):
    img = mpimg.imread(row[1]["Image Path"])
    ax = axes[i//3, i%3]
    ax.imshow(img)
    ax.axis("off")
    if row[1]["Label"] == "hem":
        ax.set_title(f"Label: hem")
    else:
        ax.set_title(f"Label: all")

plt.show()



##########################################################
###################### PIPELINE ##########################
##########################################################

batch_size = 40

train_data_generator = ImageDataGenerator(horizontal_flip=True)
valid_data_generator = ImageDataGenerator()

train_gen = train_data_generator.flow_from_dataframe(train_df, x_col="Image Path", y_col="Label", target_size=(224, 224), class_mode="categorical",
                                                     color_mode="rgb", shuffle=True, batch_size=batch_size)

valid_gen = valid_data_generator.flow_from_dataframe(valid_df, x_col="Image Path", y_col="Label", target_size=(224, 224), class_mode="categorical",
                                                     color_mode="rgb", shuffle=True, batch_size=batch_size)

test_gen = valid_data_generator.flow_from_dataframe(test_df, x_col="Image Path", y_col="Label", target_size=(224, 224), class_mode="categorical",
                                                    color_mode="rgb", shuffle=False, batch_size=batch_size)

train_steps = test_gen.n // test_gen.batch_size + 1
validation_steps = valid_gen.n // valid_gen.batch_size



##########################################################
###################### VGG16 #############################
##########################################################

img_shape=(224, 224, 3)
VGG16_base_model = VGG16(weights="imagenet", input_shape=img_shape, include_top=False, pooling=None)

VGG16_base_model.trainable = False   ## Freeze the model to keep pre-trained weights. It gives efficiency in computational costs and avoids overfitting.

last_layer = VGG16_base_model.get_layer("block5_pool")
last_output = last_layer.output
x = keras.layers.GlobalMaxPooling2D()(last_output)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(2, activation="sigmoid")(x)

VGG16_model = keras.Model(VGG16_base_model.input, x, name="VGG16_model")
VGG16_model.compile(Adamax(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
VGG16_model.summary()



##########################################################
###################### TRAINING ##########################
##########################################################

epochs = 20

history_VGG16 = VGG16_model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=valid_gen,
    validation_steps=validation_steps,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1
)



##########################################################
###################### EVALUATION ########################
##########################################################

show_history_plot(history_VGG16.history)

evaluation_matrix(VGG16_model)

show_conf_matrix(VGG16_model)







