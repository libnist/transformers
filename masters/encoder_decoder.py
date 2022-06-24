# import libraries
from keras import layers

# Define what can be imported
__all__ = ["Encoder", "Decoder"]

# The class below is a master that both Encoder
# and Decoder will get build upon it.
class Master(layers.Layer):
    def __init__(self, *, layer, layer_params,
                 number_of_layers, embedding_layer,
                 name="Master"):
        super(Master, self).__init__(name=name)

        self.layer = layer
        self.layer_params = layer_params
        self.number_of_layers = number_of_layers
        self.embedding_layer = embedding_layer
        self.name = name

        self.layers = [layer(**layer_params, name=f"{name}_{i}")
                       for i in range(number_of_layers)]

    def get_config(self):
        config = {"layer": self.layer,
                  "layer_params": self.layer_params,
                  "number_of_layers": self.number_of_layers,
                  "embedding_layer": self.embedding_layer,
                  "name": self.name}
        return config

# Class below is a Master encoder, it takes an encoder layer,
# an embedding layer and the parameters to perform making an end-to-end
# encoder.

class Encoder(layers.Layer):

    def call(self, inputs, training=False, padding_mask=None):
        # Expected input is just the expected input for embedding
        # inputs shape: (batch_size, seq_len)
        output = self.embedding_layer(inputs, training, padding_mask) # (batch_size, seq_len, d_model=d_embedding)

        for encoder in self.layers:
            output = encoder(output, training, padding_mask) # (batch_size, seq_len, d_model=d_encoder_layer)

        return output

class Decoder(layers.Layer):

    def call(self, inputs, encoder_outputs, training=False,
             padding_mask=None, look_ahead_mask=None):
        # inputs shape: (batch_size, seq_len)
        output = self.embedding_layer(inputs, training, padding_mask) # (batch_size, seq_len, d_model=d_embedding)

        for decoder in self.layers:
            output = decoder(output, encoder_outputs,
                             training, padding_mask,
                             look_ahead_mask) # (batch_size, seq_len, d_model=d_decoder_layer)
        return output
