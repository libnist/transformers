# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .components.encoder_decoder import FnetEncoder, VanillaDecoder
from .components.masks import *


class Summ(keras.Model):
    """
    An AutoEncoder based on the idea of transformers in order to compress
    long text sequences.
    """

    def __init__(
        self, *, d_model: int, number_of_layers: int,
        encoder_sequence_len: int, encoder_vocab_size: int,
        encoder_type_size: int, decoder_sequence_len: int,
        decoder_vocab_size: int, decoder_type_size: int,
        dense_layers: list, latent_dim: int, dense_dim: int = 1024,
        with_dense: bool = False, rate: float = .1, num_heads: int = 8,
        name: str = "FnetAutoEncoder"
    ) -> keras.Model:
        super(Summ, self).__init__(name=name)

        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.encoder_sequence_len = encoder_sequence_len
        self.encoder_vocab_size = encoder_vocab_size
        self.encoder_type_size = encoder_type_size
        self.decoder_sequence_len = decoder_sequence_len
        self.decoder_vocab_size = decoder_vocab_size
        self.decoder_type_size = decoder_type_size
        self.num_heads = num_heads
        self.dense_layers = dense_layers
        self.latent_dim = latent_dim
        self.dense_dim = dense_dim
        self.with_dense = with_dense
        self.rate = rate

        # Encoder Layer-start
        self.encoder = FnetEncoder(num_layers=number_of_layers,
                                   d_model=d_model,
                                   sequence_length=encoder_sequence_len,
                                   vocab_size=encoder_vocab_size,
                                   type_size=encoder_type_size,
                                   dense_dim=dense_dim,
                                   with_dense=with_dense,
                                   rate=rate)
        # Encoder Layer-end

        # Variational Auto Encoder->encoder-start
        vae_encoder_inputs = keras.Input(shape=(encoder_sequence_len, d_model),
                                         dtype=tf.float32)
        x = layers.Permute((2, 1))(vae_encoder_inputs)
        x = layers.Dense(units=encoder_sequence_len, activation="relu")(x)
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

        # Decoder Layer-start
        self.decoder = VanillaDecoder(num_layers=number_of_layers,
                                      d_model=d_model,
                                      num_heads=num_heads,
                                      sequence_length=decoder_sequence_len,
                                      vocab_size=decoder_vocab_size,
                                      type_size=decoder_type_size,
                                      dense_dim=dense_dim,
                                      rate=rate)
        # Decoder Layer-end

        # Final Layer
        self.final = layers.Dense(units=decoder_vocab_size)

    # def build(self, input_shape):
    #     input = tf.zeros(shape=(1, self.sequence_len), dtype=tf.int32)
    #     input = (input, input)
    #     enc_out = self.encoder(input)
    #     inv_enc_out = self.inv_encoder(enc_out)

    def call(self, inputs, training: bool = False, **kwargs):

        inp, tar = inputs

        _, look_ahead_mask = self.create_masks(inp=inp[0],
                                               tar=tar[0])

        encoder_outputs = self.encoder(inp, training=training)
        vae_outputs_dict = self.vae_encoder(encoder_outputs, training=training)

        # add kl loss
        if training:
            z_var = vae_outputs_dict["z_var"]
            z_mean = vae_outputs_dict["z_mean"]
            kl_loss = (-0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var)))
            self.add_loss(tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)))

        decoder_outputs = self.decoder(inputs=tar,
                                       enc_outputs=vae_outputs_dict["latent"],
                                       look_ahead_mask=look_ahead_mask,
                                       training=training)

        outputs = self.final(decoder_outputs, training=training)
        return outputs

    def train_step(self, inputs):

        inp, tar = inputs

        tar_token, tar_type = tar

        inp_tar = tar_token[:, :-1], tar_type[:, :-1]

        y_true = tar_token[:, 1:]

        with tf.GradientTape() as tape:
            predictions = self((inp, inp_tar), training=True)
            loss = self.compiled_loss(y_true=y_true,
                                      y_pred=predictions,
                                      regularization_losses=self.losses)

        vars = self.trainable_variables
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))

        self.compiled_metrics.update_state(y_true=y_true,
                                           y_pred=predictions)

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, inputs):
        
        inp, tar = inputs

        tar_token, tar_type = tar

        inp_tar = tar_token[:, :-1], tar_type[:, :-1]

        y_true = tar_token[:, 1:]

        predictions = self((inp, inp_tar), training=True)
        self.compiled_loss(y_true=y_true,
                           y_pred=predictions,
                           regularization_losses=self.losses)

        self.compiled_metrics.update_state(y_true=y_true,
                                           y_pred=predictions)

        return {metric.name: metric.result() for metric in self.metrics}

    def get_config(self):
        config = super(Summ, self).get_config()
        config.update({"d_model": self.d_model,
                       "number_of_layers": self.number_of_layers,
                       "encoder_sequence_len": self.encoder_sequence_len,
                       "encoder_vocab_size": self.encoder_vocab_size,
                       "encoder_type_size": self.encoder_type_size,
                       "decoder_sequence_len": self.decoder_sequence_len,
                       "decoder_vocab_size": self.decoder_vocab_size,
                       "decoder_type_size": self.decoder_type_size,
                       "dense_layers": self.dense_layers,
                       "num_heads": self.num_heads,
                       "latent_dim": self.latent_dim,
                       "dense_dim": self.dense_dim,
                       "with_dense": self.with_dense,
                       "rate": self.rate})
        return config

    def create_masks(self, inp, tar):
        """
        Takes in the input and target sequences w/o types and creates
        padding and lookahead mask
        """
        # Encoder padding mask (Used in the 2nd attention block in the decoder too.)
        padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        dec_target_padding_mask = tf.transpose(dec_target_padding_mask,
                                               perm=[0, 2, 1])
        look_ahead_mask = tf.minimum(dec_target_padding_mask, look_ahead_mask)
        return padding_mask, look_ahead_mask


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_var) * epsilon
