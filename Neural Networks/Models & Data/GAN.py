## Importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Dropout, Input, ReLU
from keras.optimizers import Adam

## Load and preprocess the dataset
def load_data(filename):
    data = pd.read_csv(filename)
    data.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Handle columns with string representations of numbers
    numerical_features = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
    for feature in numerical_features:
        # Convert currency and comma separated strings to float
        if data[feature].dtype == "object":
            data[feature] = data[feature].replace("[\$,]", "", regex=True).astype(float)

    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # One-hot encode categorical features
    categorical_features = ["Geography", "Gender"]
    data = pd.get_dummies(data, columns=categorical_features)

    return data, scaler

## Define the GAN
def build_gan(data_shape):
    # Generator
    generator = Sequential([
        Dense(128, input_dim=100),
        LeakyReLU(0.2),
        Dense(256),
        LeakyReLU(0.2),
        Dense(data_shape, activation="tanh")  ## it generates data within a normalized range
    ])

    # Discriminator
    discriminator = Sequential([
        Dense(256, input_dim=data_shape),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(128),
        LeakyReLU(0.2),
        Dropout(0.3),
        Dense(1, activation="sigmoid") ## it provides a probability indicating whether the input data is real or synthetic
    ])
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.7), metrics=["accuracy"])

    # Generative Adversarial Network
    discriminator.trainable = False ## it helps to train the generator more effectively
    gan_input = Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(0.0002, 0.7))

    return generator, discriminator, gan

## Training the GAN
def train_gan(generator, discriminator, gan, data, epochs=1000, batch_size=32):
    for epoch in range(epochs):
        # Random noise to synthetic data
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_data = generator.predict(noise)

        # Select a random batch of real data
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data.iloc[idx].values

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)) * 0.9)
        d_loss_fake = discriminator.train_on_batch(gen_data, np.zeros((batch_size, 1)) * 0.1)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} - D Loss: {d_loss[0]}, G Loss: {g_loss}")

def generate_and_save_synthetic_data(generator, scaler, num_samples, filename):
    # Generate random noise
    noise = np.random.normal(0, 1, (num_samples, 100))

    # Generate synthetic data
    synthetic_data = generator.predict(noise)

    synthetic_data[:,:len(scaler.scale_)] = scaler.inverse_transform(synthetic_data[:,:len(scaler.scale_)])

    # Convert to DataFrame and save as CSV
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_df.to_csv(filename, index=False)
    print(f"Synthetic data saved to {filename}")

## Main
if __name__ == "__main__":
    file_path = "C:\\Users\\Nicho\\Desktop\\Books and Labs\\ANNs\\Data\\Churn Modeling.csv"
    data, scaler = load_data(file_path)

    generator, discriminator, gan= build_gan(data.shape[1])
    train_gan(generator, discriminator, gan, data)

    # Generate and save synthetic data
    num_samples_to_generate = 100
    output_csv_file = "C:\\Users\\Nicho\\Desktop\\synthetic_data.csv"
    generate_and_save_synthetic_data(generator, scaler, num_samples_to_generate, output_csv_file)








