##### Libraries

import keras
import tensorflow as tf
from keras import layers, models, losses, optimizers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split

##### Defining the Encoder

class Encoder(layers.Layer):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.z_mean = layers.Dense(latent_dim)
        self.z_log_var = layers.Dense(latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

##### Defining the Decoder

class Decoder(layers.Layer):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.dense = layers.Dense(7*7*64, activation="relu")
        self.reshape = layers.Reshape((7, 7, 64))
        self.convT1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.convT2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.convT3 = layers.Conv2DTranspose(3, 3, activation="sigmoid", strides=(1, 1), padding="same")

    def call(self, z):
        z = self.dense(z)
        z = self.reshape(z)
        z = self.convT1(z)
        z = self.convT2(z)
        z = self.convT3(z)
        return z

##### Variational AutoEncoder (VAE) model

class VAE(models.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def call(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean ) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed

    def reparameterize(self, z_mean, z_log_var):
        eps = tf.random.normal(shape=(tf.shape(z_mean)))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

##### Dataset Preparation

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(np.asarray(img))
    return images

root_dir = r"C:\Users\Nicho\Desktop\Books and Labs\ANNs\Data\Image data"
all_images = []
labels = []

disease_types = os.listdir(root_dir)
for i, disease_types in enumerate(disease_types):
    folder_path = os.path.join(root_dir, disease_types)
    disease_images = load_images_from_folder(folder_path)
    all_images.extend(disease_images)
    labels.extend([i] * len(disease_images))

df_to_normalize = np.array(all_images)
df = df_to_normalize.astype("float32") / 255.0

##### Data Splitting

train_images, val_images = train_test_split(df, test_size=0.2, random_state=42)

##### Traning

latent_dim = 5
vae = VAE(latent_dim=latent_dim)
vae.compile(optimizer=optimizers.Adam())
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices(val_images).batch(32)
vae.fit(train_dataset, epochs=2000, validation_data=val_dataset)

##### Visual Inspection

decoder_input = keras.Input(shape=(latent_dim,))
decoder_output = Decoder(latent_dim)(decoder_input)
decoder_model = keras.Model(decoder_input, decoder_output)

def generate_images(model, n_images=5, latent_dim=latent_dim):
    latent_samples = np.random.normal(size=(n_images, latent_dim))
    generate_images = model.predict(latent_samples)
    return generate_images

generate_images = generate_images(decoder_model, 10)

plt.figure(figsize=(20, 4))
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(generate_images[i].reshape(28, 28, 3))
    plt.gray()
    ax.axis("off")
plt.show()

