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

        # self.ln = layers.Normalization()

        # self.build(None)

    def build(self, input_shape):
        input = tf.zeros(shape=(1, self.sequence_len), dtype=tf.int32)
        input = (input, input)
        enc_out = self.encoder(input)
        inv_enc_out = self.inv_encoder(enc_out)

    def call(self, inputs, training: bool = False, **kwargs):
        encoder_outputs, embeddings = self.encoder(inputs,
                                                   with_embeddings=True,
                                                   training=training)
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
        return inv_encoder_outputs, embeddings

    def compile(self, mlm_optim=None,
                vae_optim=None,
                mlm_loss=None,
                vae_loss=None,
                mlm_accuracy=None,
                **kwargs):  # Have to put proper default values
        super(FnetAutoEncoder, self).compile(**kwargs)

        self.mlm_optim = mlm_optim
        self.vae_optim = vae_optim
        self.mlm_loss = mlm_loss
        self.vae_loss = vae_loss

        self.mlm_accuracy = mlm_accuracy

        self.mlm_loss_metric = keras.metrics.Mean(name="MLM_Loss")
        self.vae_loss_metric = keras.metrics.Mean(name="VAE_Loss")

    def mlm_step(self, inputs, training=True):
        encoder_outputs = self.encoder(inputs, training=True)
        mlm_outputs = self.mlm_model(encoder_outputs, training=True)
        return mlm_outputs

    def train_step(self, inputs):
        inputs, labels = inputs

        # Masked Language step
        with tf.GradientTape() as tape:
            mlm_predictions = self.mlm_step(inputs)
            mlm_loss = self.mlm_loss(y_true=labels,
                                     y_pred=mlm_predictions)

        mlm_vars = (self.encoder.trainable_variables +
                    self.mlm_model.trainable_variables)
        grads = tape.gradient(mlm_loss, mlm_vars)
        self.mlm_optim.apply_gradients(zip(grads, mlm_vars))

        # VAE step
        with tf.GradientTape() as tape:
            predictions, embeddings = self((labels, inputs[1]), training=True)
            vae_loss = self.vae_loss(y_true=embeddings,
                                     y_pred=predictions)
            vae_loss += self.losses

        vae_vars = (self.encoder.trainable_variables +
                    self.vae_encoder.trainable_variables +
                    self.vae_decoder.trainable_variables +
                    self.inv_encoder.trainable_variables)
        grads = tape.gradient(vae_loss, vae_vars)
        self.vae_optim.apply_gradients(zip(grads, vae_vars))

        self.mlm_loss_metric.update_state(mlm_loss)
        self.vae_loss_metric.update_state(vae_loss)

        self.mlm_accuracy.update_state(y_true=labels,
                                       y_pred=mlm_predictions)

        return {metric.name: metric.result() for metric in self.metrics}

    @property
    def metrics(self):
        return [self.mlm_loss_metric,
                self.mlm_accuracy,
                self.vae_loss_metric]

    def test_step(self, inputs):
        inputs, labels = inputs

        # Test MLM
        mlm_predictions = self.mlm_step(inputs, training=False)
        mlm_loss = self.mlm_loss(y_true=labels,
                                 y_pred=mlm_predictions)

        # Test VAE
        predictions, embeddings = self((labels, inputs[1]), training=False)
        vae_loss = self.vae_loss(y_true=embeddings,
                                 y_pred=predictions)

        self.mlm_loss_metric.update_state(mlm_loss)
        self.vae_loss_metric.update_state(vae_loss)

        self.mlm_accuracy.update_state(y_true=labels,
                                       y_pred=mlm_predictions)

        return {metric.name: metric.result() for metric in self.metrics}

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
