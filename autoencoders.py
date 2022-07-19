# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .components.encoder_decoder import FnetEncoder, InverseFnetEncoder


class FnetAutoEncoder(keras.Model):
    """
    An AutoEncoder based on the idea of transformers in order to compress
    long text sequences.
    """

    def __init__(
        self, *, d_model: int, number_of_layers: int, sequence_len: int,
        vocab_size: int, type_size: int, dense_layers: list, latent_dim: int,
        dense_dim: int = 1024, with_dense: bool = False, rate: float = .1,
        name: str = "FnetAutoEncoder"
    ) -> keras.Model:
        super(FnetAutoEncoder, self).__init__(name=name)

        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.dense_layers = dense_layers
        self.latent_dim = latent_dim
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        # Encoder Layer-start
        self.encoder = FnetEncoder(num_layers=number_of_layers,
                                   d_model=d_model,
                                   sequence_length=sequence_len,
                                   vocab_size=vocab_size,
                                   type_size=type_size,
                                   dense_dim=dense_dim,
                                   with_dense=with_dense,
                                   rate=rate)
        # Encoder Layer-end

        # Variational Auto Encoder-start
        # Variational Auto Encoder->encoder-start
        vae_encoder_inputs = keras.Input(shape=(sequence_len, d_model),
                                         dtype=tf.float32)
        x = layers.Permute((2, 1))(vae_encoder_inputs)
        x = layers.Dense(units=sequence_len, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)

        for units in dense_layers:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        z_mean = layers.Dense(units=latent_dim, name="z_mean")(x)
        z_var = layers.Dense(units=latent_dim, name="z_var")(x)
        z = Sampling()([z_mean, z_var])
        vae_encoder_outputs = layers.Permute((2, 1))(z)
        self.vae_encoder = keras.Model(inputs=vae_encoder_inputs,
                                       outputs={"z_mean": z_mean,
                                                "z_var": z_var,
                                                "latent": vae_encoder_outputs},
                                       name="VAEEncoder")
        # Variational Auto Encoder->encoder-end

        # Variational Auto Encoder->decoder-start
        vae_decoder_inputs = keras.Input(shape=(latent_dim, d_model),
                                         dtype=tf.float32)
        x = layers.Permute((2, 1))(vae_decoder_inputs)

        for units in dense_layers[::-1]:
            x = layers.Dense(units=units, activation="relu")(x)
            x = layers.Dropout(rate=rate)(x)

        x = layers.Dense(units=sequence_len, activation="relu")(x)
        x = layers.Dropout(rate=rate)(x)
        vae_decoder_outputs = layers.Permute((2, 1))(x)

        self.vae_decoder = keras.Model(inputs=vae_decoder_inputs,
                                       outputs=vae_decoder_outputs,
                                       name="VAEDecoder")
        # Variational Auto Encoder->decoder-end
        # Variational Auto Encoder-end

        # Inverse Encoder-start
        self.inv_encoder = InverseFnetEncoder(num_layers=number_of_layers,
                                              d_model=d_model,
                                              dense_dim=dense_dim,
                                              with_dense=with_dense,
                                              rate=rate)
        # Inverse Encoder-end

        # Masked Language Model-start
        self.mlm_model = keras.Sequential(
            [keras.Input(shape=(sequence_len, d_model)),
             layers.Dense(units=vocab_size)],
            name="MaskedLanguageModel"
        )
        # Masked Language Model-end

        # self.build(None)

    def build(self, input_shape):
        input = tf.zeros(shape=(1, self.sequence_len), dtype=tf.int32)
        input = (input, input)
        enc_out = self.encoder(input)
        inv_enc_out = self.inv_encoder(enc_out)

    def call(self, inputs, training: bool = False, ** kwargs):
        encoder_outputs = self.encoder(inputs, training=training)
        vae_outputs_dict = self.vae_encoder(encoder_outputs, training=training)
        # add kl loss
        if training:
            z_var = vae_outputs_dict["z_var"]
            z_mean = vae_outputs_dict["z_mean"]
            kl_loss = (-0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var)))
            self.add_loss(tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))
        vae_outputs = self.vae_decoder(vae_outputs_dict["latent"],
                                       training=training)
        inv_encoder_outputs = self.inv_encoder(vae_outputs, training=training)
        mlm_outputs = self.mlm_model(inv_encoder_outputs, training=training)
        return mlm_outputs

    def train_step(self, inputs):

        # This is not for masked modeling i will change this later

        with tf.GradientTape() as tape:
            prediction = self(inputs, training=True)
            loss = self.compiled_loss(y_true=inputs[0],
                                      y_pred=prediction,
                                      regularization_losses=self.losses)

        vars = self.trainable_variables
        grads = tape.gradient(loss, vars)

        self.optimizer.apply_gradients(zip(grads, vars))

        self.compiled_metrics.update_state(y_true=inputs[0],
                                           y_pred=prediction,)

        return {metric.name: metric.result()
                for metric in self.metrics}

    def get_config(self):
        config = super(FnetAutoEncoder, self).get_config()
        config.update({"d_model": self.d_model,
                       "number_of_layers": self.number_of_layers,
                       "sequence_len": self.sequence_len,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "dense_layers": self.dense_layers,
                       "latent_dim": self.latent_dim,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
