from flask import Flask, request, render_template, jsonify, send_file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
import io
import base64
import json
import os
from datetime import datetime
import plotly.graph_objs as go
import plotly.utils
from tensorflow.keras import layers, Model, backend as K
from tensorflow.keras.optimizers.legacy import Adam
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store model and scaler
trained_model = None
scaler = None
training_data = None
feature_names = None

import logging
from tensorflow.keras import layers, Model, backend as K
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, backend as K

import math
import numpy as np

# ---- PSO Hyperparameters ----
DIMENSIONS = 4            # Number of dimensions (start with 2, will increase for VAE+GAN)
GLOBAL_BEST = 0             # Best objective value (minimization)
B_LO = -5                   # Lower boundary of search space
B_HI = 5                    # Upper boundary of search space

POPULATION = 20             # Population count (n_pop)
V_MAX = 0.1                 # Maximum velocity value (v_max)
PERSONAL_C = 2.0            # Personal coefficient factor (c1)
SOCIAL_C = 2.0              # Social coefficient factor (c2)
CONVERGENCE = 0.001         # Convergence value for stopping
MAX_ITER = 10              # Maximum number of iterations (max_iter)

def gan_vae_fitness(params, X_train, X_val, input_dim):
    # Unpack your hyperparameters from params list
    latent_dim = int(params[0])
    ae_hidden_dim = int(params[1])
    d_hidden_dim = int(params[2])
    kl_beta = float(params[3])

    # Build and train a model for a few epochs to evaluate fitness
    model = AdversarialAutoencoderWithDualDiscriminator(
        input_dim=input_dim,
        latent_dim=latent_dim,
        ae_hidden_dim=ae_hidden_dim,
        d_hidden_dim=d_hidden_dim,
        kl_beta=kl_beta,
        adv_weight_latent=1.0,
        adv_weight_data=1.0
    )
    model.compile(
        enc_dec_optimizer=Adam(learning_rate=0.001),
        latent_disc_optimizer=Adam(learning_rate=0.001),
        data_disc_optimizer=Adam(learning_rate=0.001)
    )

    # Train for a small number of epochs for speed (e.g., 5)
    history = model.fit(
        X_train, epochs=5, batch_size=32, validation_data=(X_val,), verbose=0
    )

    # Use the validation recon/discriminator losses to define fitness:
    recon_loss = np.mean(history.history['val_recon_loss'][-2:])
    ld_loss    = np.mean(history.history['val_ld_loss'][-2:])
    xd_loss    = np.mean(history.history['val_xd_loss'][-2:])
    # You can also include other losses for more nuanced fitness

    # Combine as a weighted sum exactly as the EvoAAE paper suggests
    fitness = recon_loss + ld_loss + xd_loss
    return fitness
def pso_optimize_vae_gan(X_train, X_val, input_dim):
    # (Define bounds for each hyperparameter you want to optimize)
    param_bounds = [
        (4, 32),     # latent_dim
        (16, 128),   # ae_hidden_dim
        (16, 128),   # d_hidden_dim
        (0.5, 3.0),  # kl_beta
    ]
    pop_size = 8
    max_iter = 25
    print(f"Starting PSO optimization with {pop_size} particles for {max_iter} iterations...")
    v_max = 8

    # Initialize particles
    swarm = []
    for _ in range(pop_size):
        init = [np.random.uniform(low, high) for (low, high) in param_bounds]
        velocity = [np.random.uniform(-v_max, v_max) for _ in init]
        fitness = gan_vae_fitness(init, X_train, X_val, input_dim)
        swarm.append({
            "position": init,
            "velocity": velocity,
            "fitness": fitness,
            "best_pos": init[:],
            "best_fitness": fitness,
        })
    # Global best
    best = min(swarm, key=lambda x: x['fitness'])
    g_best_pos = best["position"][:]
    g_best_fitness = best["fitness"]

    for iteration in range(max_iter):
        #log each iteration and see what result we get for the pso
        print(f"Iteration {iteration + 1}/{max_iter} - Global Best Fitness: {g_best_fitness:.4f}")
        for p in swarm:
            # PSO update
            for i, (low, high) in enumerate(param_bounds):
                r1, r2 = np.random.rand(), np.random.rand()
                p["velocity"][i] = (
                    0.7 * p["velocity"][i]
                    + PERSONAL_C * r1 * (p["best_pos"][i] - p["position"][i])
                    + SOCIAL_C * r2 * (g_best_pos[i] - p["position"][i])
                )
                p["velocity"][i] = np.clip(p["velocity"][i], -v_max, v_max)
                p["position"][i] = np.clip(
                    p["position"][i] + p["velocity"][i], low, high
                )

            # Evaluate
            p["fitness"] = gan_vae_fitness(p["position"], X_train, X_val, input_dim)
            if p["fitness"] < p["best_fitness"]:
                p["best_pos"] = p["position"][:]
                p["best_fitness"] = p["fitness"]
                if p["fitness"] < g_best_fitness:
                    g_best_pos = p["position"][:]
                    g_best_fitness = p["fitness"]

    return g_best_pos, g_best_fitness

class AdversarialAutoencoderWithDualDiscriminator(keras.Model):
    def __init__(self, input_dim, latent_dim=8, ae_hidden_dim=32, d_hidden_dim=32, kl_beta=1.0,
                 adv_weight_latent=1.0, adv_weight_data=1.0):
        super().__init__()
        # Encoder
        self.encoder_input = layers.Input(shape=(input_dim,))
        x = layers.Dense(ae_hidden_dim, activation='relu')(self.encoder_input)
        self.z_mean = layers.Dense(latent_dim, name='z_mean')(x)
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
        self.z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.encoder_input, [self.z_mean, self.z_log_var, self.z], name='encoder')

        # Decoder
        self.decoder_input = layers.Input(shape=(latent_dim,))
        d = layers.Dense(ae_hidden_dim, activation='relu')(self.decoder_input)
        x_rec = layers.Dense(input_dim, activation='linear')(d)
        self.decoder = Model(self.decoder_input, x_rec, name='decoder')

        # Discriminator 1: in latent (z) space
        self.latent_disc_input = layers.Input(shape=(latent_dim,))
        h1 = layers.Dense(d_hidden_dim, activation='relu')(self.latent_disc_input)
        z_out = layers.Dense(1, activation='sigmoid')(h1)
        self.latent_discriminator = Model(self.latent_disc_input, z_out, name='latent_discriminator')

        # Discriminator 2: on data/reconstruction space
        self.data_disc_input = layers.Input(shape=(input_dim,))
        h2 = layers.Dense(d_hidden_dim, activation='relu')(self.data_disc_input)
        x_out = layers.Dense(1, activation='sigmoid')(h2)
        self.data_discriminator = Model(self.data_disc_input, x_out, name='data_discriminator')

        # Loss weights
        self.kl_beta = kl_beta
        self.adv_weight_latent = adv_weight_latent
        self.adv_weight_data = adv_weight_data

        # Loss trackers for Keras
        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.ld_loss_tracker = keras.metrics.Mean(name="ld_loss")
        self.xd_loss_tracker = keras.metrics.Mean(name="xd_loss")
        self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    @property
    def metrics(self):
        # List the tracked metrics
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.ld_loss_tracker,
            self.xd_loss_tracker,
            self.adv_loss_tracker,
        ]

    def compile(self, enc_dec_optimizer, latent_disc_optimizer, data_disc_optimizer, **kwargs):
        super().compile(**kwargs)
        self.enc_dec_optimizer = enc_dec_optimizer
        self.latent_disc_optimizer = latent_disc_optimizer
        self.data_disc_optimizer = data_disc_optimizer

        self.latent_discriminator.compile(optimizer=latent_disc_optimizer, loss='binary_crossentropy')
        self.data_discriminator.compile(optimizer=data_disc_optimizer, loss='binary_crossentropy')


    def train_step(self, data):
        X = data if isinstance(data, tf.Tensor) else data[0]
        batch_size = tf.shape(X)[0]
        # 1. Forward VAE (encode, sample, decode)
        z_mean, z_log_var, z = self.encoder(X)
        X_rec = self.decoder(z)

        # 2. VAE reconstruction+KL loss, train enc+dec
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(X)
            X_rec = self.decoder(z)
            rec_loss = tf.reduce_mean(tf.square(X - X_rec))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            vae_loss = rec_loss + self.kl_beta * kl_loss
        grads = tape.gradient(vae_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)
        n_enc = len(self.encoder.trainable_variables)
        self.enc_dec_optimizer.apply_gradients(zip(grads[:n_enc], self.encoder.trainable_variables))
        self.enc_dec_optimizer.apply_gradients(zip(grads[n_enc:], self.decoder.trainable_variables))

        # 3. Latent space discriminator (real z: ~N(0,1); fake z: encoder)
        z_real = tf.random.normal(shape=(batch_size, z.shape[1]))
        valid = tf.ones((batch_size, 1))
        fake = tf.zeros((batch_size, 1))
        # Train latent discriminator manually
        with tf.GradientTape() as tape_ld:
            ld_real_pred = self.latent_discriminator(z_real)
            ld_fake_pred = self.latent_discriminator(z)
            ld_loss_real = tf.keras.losses.binary_crossentropy(valid, ld_real_pred)
            ld_loss_fake = tf.keras.losses.binary_crossentropy(fake, ld_fake_pred)
            ld_loss = 0.5 * (tf.reduce_mean(ld_loss_real) + tf.reduce_mean(ld_loss_fake))

        ld_grads = tape_ld.gradient(ld_loss, self.latent_discriminator.trainable_variables)
        self.latent_disc_optimizer.apply_gradients(zip(ld_grads, self.latent_discriminator.trainable_variables))


        # 4. Data space discriminator (real X; fake X')
        # Train data discriminator manually
        with tf.GradientTape() as tape_xd:
            xd_real_pred = self.data_discriminator(X)
            xd_fake_pred = self.data_discriminator(X_rec)
            xd_loss_real = tf.keras.losses.binary_crossentropy(valid, xd_real_pred)
            xd_loss_fake = tf.keras.losses.binary_crossentropy(fake, xd_fake_pred)
            xd_loss = 0.5 * (tf.reduce_mean(xd_loss_real) + tf.reduce_mean(xd_loss_fake))

        xd_grads = tape_xd.gradient(xd_loss, self.data_discriminator.trainable_variables)
        self.data_disc_optimizer.apply_gradients(zip(xd_grads, self.data_discriminator.trainable_variables))


        # 5. Encoder + decoder update to fool discriminators
        with tf.GradientTape() as tape_adv:
            z_mean, z_log_var, z = self.encoder(X)
            X_rec = self.decoder(z)
            ld_pred = self.latent_discriminator(z)
            xd_pred = self.data_discriminator(X_rec)
            adv_loss_z = tf.keras.losses.binary_crossentropy(valid, ld_pred)
            adv_loss_x = tf.keras.losses.binary_crossentropy(valid, xd_pred)
            adv_loss = self.adv_weight_latent * tf.reduce_mean(adv_loss_z) + self.adv_weight_data * tf.reduce_mean(adv_loss_x)
        adv_grads = tape_adv.gradient(adv_loss, self.encoder.trainable_variables)
        self.enc_dec_optimizer.apply_gradients(zip(adv_grads, self.encoder.trainable_variables))

        total_loss = vae_loss + adv_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.ld_loss_tracker.update_state(ld_loss)
        self.xd_loss_tracker.update_state(xd_loss)
        self.adv_loss_tracker.update_state(adv_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "ld_loss": self.ld_loss_tracker.result(),
            "xd_loss": self.xd_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
        }

    def test_step(self, data):
        X = data[0] if isinstance(data, tuple) else data

        # Forward pass
        z_mean, z_log_var, z = self.encoder(X)
        X_rec = self.decoder(z)

        # Same losses as in train_step, but NO optimizer
        rec_loss = tf.reduce_mean(tf.square(X - X_rec))
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))

        # Latent discriminator loss (on sampled z)
        z_real = tf.random.normal(shape=(tf.shape(X)[0], tf.shape(z)[1]))
        valid = tf.ones((tf.shape(X)[0], 1))
        fake = tf.zeros((tf.shape(X)[0], 1))
        ld_real_pred = self.latent_discriminator(z_real, training=False)
        ld_fake_pred = self.latent_discriminator(z, training=False)
        ld_loss_real = tf.keras.losses.binary_crossentropy(valid, ld_real_pred)
        ld_loss_fake = tf.keras.losses.binary_crossentropy(fake, ld_fake_pred)
        ld_loss = 0.5 * (tf.reduce_mean(ld_loss_real) + tf.reduce_mean(ld_loss_fake))

        # Data discriminator loss (on reconstructions)
        xd_real_pred = self.data_discriminator(X, training=False)
        xd_fake_pred = self.data_discriminator(X_rec, training=False)
        xd_loss_real = tf.keras.losses.binary_crossentropy(valid, xd_real_pred)
        xd_loss_fake = tf.keras.losses.binary_crossentropy(fake, xd_fake_pred)
        xd_loss = 0.5 * (tf.reduce_mean(xd_loss_real) + tf.reduce_mean(xd_loss_fake))

        # Adversarial loss for generator (optional; unused in validation, but you can average it too if you want)
        adv_loss_z = tf.keras.losses.binary_crossentropy(valid, self.latent_discriminator(z, training=False))
        adv_loss_x = tf.keras.losses.binary_crossentropy(valid, self.data_discriminator(X_rec, training=False))
        adv_loss = self.adv_weight_latent * tf.reduce_mean(adv_loss_z) + self.adv_weight_data * tf.reduce_mean(adv_loss_x)

        total_loss = rec_loss + self.kl_beta * kl_loss + adv_loss

        # Update metrics (these are what Keras will log in history.history, as val_* when running validation step)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.ld_loss_tracker.update_state(ld_loss)
        self.xd_loss_tracker.update_state(xd_loss)
        self.adv_loss_tracker.update_state(adv_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "ld_loss": self.ld_loss_tracker.result(),
            "xd_loss": self.xd_loss_tracker.result(),
            "adv_loss": self.adv_loss_tracker.result(),
        }

    def compute_reconstruction_error(self, X):
        z_mean, z_log_var, z = self.encoder.predict(X)
        X_rec = self.decoder.predict(z)
        mse = np.mean((X - X_rec) ** 2, axis=1)
        return mse
    def call(self, inputs, training=False):
        """Forward pass used by model.fit, model.predict, etc."""
        _, _, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction





class VariationalAutoencoder(Model):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32):
        super().__init__()
        # Encoder
        self.encoder_inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(hidden_dim, activation='relu')(self.encoder_inputs)
        self.z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self.z = layers.Lambda(self.sampling, output_shape=(latent_dim,), name='z')([self.z_mean, self.z_log_var])
        self.encoder = Model(self.encoder_inputs, [self.z_mean, self.z_log_var, self.z], name="encoder")

        # Decoder
        self.decoder_inputs = layers.Input(shape=(latent_dim,))
        d = layers.Dense(hidden_dim, activation='relu')(self.decoder_inputs)
        decoder_outputs = layers.Dense(input_dim, activation='linear')(d)
        self.decoder = Model(self.decoder_inputs, decoder_outputs, name="decoder")

        # VAE model
        outputs = self.decoder(self.encoder(self.encoder_inputs)[2])
        super().__init__(self.encoder_inputs, outputs)
        self.total_loss_tracker = keras.metrics.Mean(name="loss")

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Calculate the losses
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(tf.square(data - reconstruction), axis=1))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {"loss": self.total_loss_tracker.result()}

    def compute_reconstruction_error(self, X):
        z_mean, z_log_var, z = self.encoder.predict(X)
        X_pred = self.decoder.predict(z)
        mse = np.mean((X - X_pred) ** 2, axis=1)
        return mse


class BasicAutoencoder:
    """
    PLACEHOLDER: Basic Autoencoder for anomaly detection
    TODO: Replace this with EvoAAE implementation (VAE + GAN + PSO)
    """
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.build_model()
    
    def build_model(self):
        # Encoder
        input_layer = keras.Input(shape=(self.input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(optimizer='adam', loss='mse')
    
    def train(self, X_train, epochs=50, batch_size=32):
        history = self.model.fit(X_train, X_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_split=0.2,
                                shuffle=True,
                                verbose=1)
        return history
    
    def predict(self, X):
        return self.model.predict(X)
    
    def anomaly_score(self, X):
        reconstructed = self.predict(X)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        return mse

def validate_data_format(df):
    """Validate uploaded data format"""
    issues = []
    
    # Check if DataFrame is empty
    if df.empty:
        issues.append("The uploaded file is empty.")
        return issues
    
    # Check for minimum columns
    if len(df.columns) < 2:
        issues.append("Data should have at least 2 columns (timestamp + at least 1 feature).")
    
    # Check for timestamp column
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if not timestamp_cols:
        issues.append("No timestamp column found. Please include a column with 'time' or 'date' in the name.")
    
    # Check for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 1:
        issues.append("At least one numeric column is required for analysis.")
    
    # Check for missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 50]
    if not high_missing.empty:
        issues.append(f"Columns with >50% missing values: {high_missing.index.tolist()}")
    
    return issues

def preprocess_data(df):
    """Preprocess the uploaded data"""
    global feature_names
    
    # Identify timestamp column
    timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if timestamp_cols:
        timestamp_col = timestamp_cols[0]
        # Try to parse timestamp
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            df = df.sort_values(timestamp_col)
        except:
            pass
    
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_names = numeric_cols
    
    # Fill missing values with forward fill then backward fill
    df_numeric = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaN rows
    df_numeric = df_numeric.dropna()
    
    return df_numeric, timestamp_cols[0] if timestamp_cols else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global trained_model, scaler, training_data, feature_names
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format. Please upload CSV or Excel files.'}), 400
        
        # Validate data format
        issues = validate_data_format(df)
        if issues:
            return jsonify({'error': 'Data validation failed', 'issues': issues}), 400
        
        # Preprocess data
        df_processed, timestamp_col = preprocess_data(df)
        
        # Store training data
        training_data = df_processed
        
        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_processed)
        
        # Train the model (PLACEHOLDER: Basic Autoencoder)
        # TODO: Replace with EvoAAE (VAE + GAN + PSO)
        input_dim = X_scaled.shape[1]
        # trained_model = VariationalAutoencoder(input_dim=input_dim)
        # trained_model = AdversarialAutoencoderWithDualDiscriminator(input_dim=input_dim,latent_dim=8, ae_hidden_dim=32, d_hidden_dim=32, kl_beta=1.0,
        #          adv_weight_latent=1.0, adv_weight_data=1.0)
        # trained_model.compile(enc_dec_optimizer=Adam(learning_rate=0.001),
        #                       latent_disc_optimizer=Adam(learning_rate=0.001),
        #                       data_disc_optimizer=Adam(learning_rate=0.001))
        # # trained_model.compile(optimizer='adam')
        # # history = trained_model.fit(X_scaled, X_scaled, epochs=50, batch_size=32, verbose=1)
        # # Train the model
        # # history = trained_model.train(X_scaled, epochs=50)
        # #make validation data 20% of X_scaled
        # X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        # history = trained_model.fit(X_train, epochs=50, batch_size=32, validation_data=(X_val,))

        # Preprocess, scale
        
        X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
        # === Start of PSO Hyperparameter Search ===
        best_pos, best_fitness = pso_optimize_vae_gan(X_train, X_val, input_dim)
        # best_pos = [latent_dim, ae_hidden_dim, d_hidden_dim, kl_beta, ...]

        # Final model with best hyperparams
        trained_model = AdversarialAutoencoderWithDualDiscriminator(
            input_dim=X_train.shape[1],
            latent_dim=int(best_pos[0]),
            ae_hidden_dim=int(best_pos[1]),
            d_hidden_dim=int(best_pos[2]),
            kl_beta=best_pos[3],
        )
        trained_model.compile(
            enc_dec_optimizer=Adam(learning_rate=0.001),
            latent_disc_optimizer=Adam(learning_rate=0.001),
            data_disc_optimizer=Adam(learning_rate=0.001)
        )
        history = trained_model.fit(X_train, epochs=50, batch_size=32, validation_data=(X_val,))
        # Generate data summary
        summary = {
            'total_samples': len(df_processed),
            'features': feature_names,
            'num_features': len(feature_names),
            'timestamp_column': timestamp_col,
            'training_loss': float(training_loss := history.history['loss'][-1] if 'loss' in history.history else None),
            'validation_loss': float(validation_loss := history.history['val_loss'][-1] if 'val_loss' in history.history else None)
        }
        
        return jsonify({
            'message': 'Model trained successfully!',
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/detect', methods=['POST'])
def detect_anomalies():
    global trained_model, scaler, feature_names
    
    if trained_model is None:
        return jsonify({'error': 'No trained model available. Please upload training data first.'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Preprocess data
        df_processed, timestamp_col = preprocess_data(df)
        
        # Ensure same features as training data
        if not all(col in df_processed.columns for col in feature_names):
            return jsonify({'error': 'Feature mismatch. Upload data must have the same features as training data.'}), 400
        
        df_processed = df_processed[feature_names]
        
        # Normalize data using the same scaler
        X_scaled = scaler.transform(df_processed)
        
        # Calculate anomaly scores
        anomaly_scores = trained_model.compute_reconstruction_error(X_scaled)
        
        # Determine threshold (95th percentile of training data scores)
        training_scaled = scaler.transform(training_data)
        training_scores = trained_model.compute_reconstruction_error(training_scaled)
        threshold = np.percentile(training_scores, 95)
        
        # Flag anomalies
        anomalies = anomaly_scores > threshold
        
        # Create results DataFrame
        results_df = df_processed.copy()
        results_df['anomaly_score'] = anomaly_scores
        results_df['is_anomaly'] = anomalies
        results_df['timestamp'] = range(len(results_df))  # Use index as timestamp if no timestamp column
        
        if timestamp_col and timestamp_col in df.columns:
            results_df['timestamp'] = df[timestamp_col].values
        
        # Create visualization data
        viz_data = {
            'timestamps': results_df['timestamp'].astype(str).tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomalies': anomalies.tolist(),
            'threshold': float(threshold)
        }
        
        # Summary statistics
        summary = {
            'total_samples': len(results_df),
            'anomalies_detected': int(np.sum(anomalies)),
            'anomaly_percentage': float(np.mean(anomalies) * 100),
            'threshold': float(threshold),
            'max_anomaly_score': float(np.max(anomaly_scores)),
            'avg_anomaly_score': float(np.mean(anomaly_scores))
        }
        
        return jsonify({
            'message': 'Anomaly detection completed!',
            'summary': summary,
            'visualization': viz_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Error during anomaly detection: {str(e)}'}), 500


@app.route('/pso_optimize', methods=['POST'])
def pso_optimize():
    # (Optionally: allow passing bounds or PSO settings by POST body)
    try:
        # For now, just run the standard PSO (Ackley) routine to prove it works.
        best_pos, best_fitness = particle_swarm_optimization()
        return jsonify({
            'best_position': best_pos,
            'best_fitness': best_fitness
        })
    except Exception as e:
        return jsonify({'error': f'PSO error: {str(e)}'})

@app.route('/data_format_info')
def data_format_info():
    format_info = {
        'title': 'Data Format Requirements',
        'requirements': [
            'File format: CSV or Excel (.xlsx, .xls)',
            'Must contain at least 2 columns',
            'Include a timestamp column (column name should contain "time" or "date")',
            'At least one numeric feature column for analysis',
            'Data should represent normal operating conditions for training',
            'Missing values should be minimal (<50% per column)',
            'Multivariate time series data is supported'
        ],
        'example_structure': {
            'timestamp': ['2023-01-01 00:00:00', '2023-01-01 00:01:00', '2023-01-01 00:02:00'],
            'temperature': [25.5, 26.1, 25.8],
            'pressure': [1013.2, 1013.5, 1013.1],
            'flow_rate': [100.5, 101.2, 99.8]
        }
    }
    return jsonify(format_info)

if __name__ == '__main__':
    app.run(debug=True)