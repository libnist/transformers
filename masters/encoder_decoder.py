# Import libraries
from keras import layers

# __all__ defines what can be imported
__all__ = ["Encoder", "Decoder"]


class Master(layers.Layer):
    """
    High level encoder and decoder layers inherit this class and just implement
    their own `call()` method.
    """

    def __init__(
        self, *, layer: layers.Layer, number_of_layers: int,
        embedding_layer: layers.Layer, name: str = "Master"
    ) -> layers.Layer:
        super().__init__(name=name)

        self.layer = layer
        self.number_of_layers = number_of_layers
        self.embedding_layer = embedding_layer

        layer_config = layer.get_config()
        layer_name = layer_config["name"]

        self.layers = []

        for i in range(number_of_layers):
            layer_config.update(name=f"{layer_name}_{i}")
            self.layers.append(layer.from_config(layer_config))

    def get_config(self):
        config = super().get_config()
        config.update({"layer": self.layer,
                       "number_of_layers": self.number_of_layers,
                       "embedding_layer": self.embedding_layer})
        return config

# Class below is a Master encoder, it takes an encoder layer,
# an embedding layer and the parameters to perform making an end-to-end
# encoder.


class Encoder(Master):
    """
    A high level Encoder that given an encoder layer and its embedding layer
    first embeds the inputs and than passes the embedded tokens through
    `number_of_layers` encoder layers. the output of the last encoder layer
    will be returned as the output.
    """
    def call(self, inputs, training: bool = False, padding_mask=None):
        # Expected input is just the expected input for embedding
        # inputs shape: (batch_size, seq_len)
        # (batch_size, seq_len, d_model=d_embedding)
        output = self.embedding_layer(inputs=inputs, training=training)

        for encoder in self.layers:
            # (batch_size, seq_len, d_model=d_encoder_layer)
            output = encoder(inputs=output,
                             training=training,
                             padding_mask=padding_mask)

        return output


class Decoder(Master):
    """
    A high level Decoder that given a decoder layer and its embedding layer
    first embeds the inputs and than passes the embedded tokens through
    `number_of_layers` encoder layers. the output of the last decoder layer
    will be returned as the output.
    """
    def call(self, inputs, enc_outputs, training: bool = False,
             padding_mask=None, look_ahead_mask=None):
        # inputs shape: (batch_size, seq_len)
        # (batch_size, seq_len, d_model=d_embedding)
        output = self.embedding_layer(inputs, training)

        for decoder in self.layers:
            # (batch_size, seq_len, d_model=d_decoder_layer)
            output = decoder(inputs=output, enc_outputs=enc_outputs,
                             training=training, padding_mask=padding_mask,
                             look_ahead_mask=look_ahead_mask)
        return output
