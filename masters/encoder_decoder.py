# import libraries
import tensorflow as tf
import keras

from keras import layers

# Class below is a Master encoder, it takes an encoder layer,
# an embedding layer and the parameters to perform making an end-to-end
# encoder.

class Encoder(layers.Layer):

    def __init__(self, *, encoder_layer, encoder_params,
                 number_of_layers, embedding_layer, name="EncoderMaster"):
        super(Encoder, self).__init__(name=name)

        self.embedding_layer = embedding_layer

        self.encoder_layer = encoder_layer
        self.encoder_params = encoder_params
        self.number_of_layers = number_of_layers

        self.name = name
        

        self.encoders = [encoder_layer(**encoder_params, name=f"encoder_layer_{layer}")
                         for layer in range(number_of_layers)]

    def call(self, inputs, training=False, padding_mask=None):
        # inputs shape: (batch_size, seq_len)
        output = self.embedding_layer(inputs, training, padding_mask) # (batch_size, seq_len, d_model=d_embedding)

        for encoder in self.encoders:
            output = encoder(output, training, padding_mask) # (batch_size, seq_len, d_model=d_encoder_layer)

        return output

    def get_config(self):
        config = {"encoder_layer": self.encoder_layer,
                  "encoder_params": self.encoder_params,
                  "number_of_layers": self.number_of_layers,
                  "embedding_layer": self.embedding_layer,
                  "name": self.name}
        return config
