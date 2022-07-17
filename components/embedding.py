# Import libraries
import tensorflow as tf
from keras import layers

# __all__ defines the things that can be accessed from outside
__all__ = ["EmbeddingLayer"]


# This Module contains an embedding layer that is the same as
# the one used in bert. it has three types of embeddings, positional,
# token, and type embedding. it returns the net sum of all the embeddings.


class EmbeddingLayer(layers.Layer):
    """
    This embedding layer is designed to use token embeddings,
    positional embeddings, and type embeddings. `d_model` defines the output
    dimention of the embedding layer. there is also a `rate` parameter in order
    to use a dropout layer just after when embedded outputs are achived.
    sequence_length: used to delimite positional embedding.
    vocab_size: your vocabulary size or number of tokens.
    type_size: used to delimite type embedding (for sentences).
    """

    def __init__(
        self, *, sequence_length: int, vocab_size: int, type_size: int,
        d_model: int, rate: float = .1, name: str = "EmbeddingLayer", **kwargs
    ) -> layers.Layer:
        super(EmbeddingLayer, self).__init__(name=name, **kwargs)

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.type_size = type_size
        self.d_model = d_model
        self.rate = rate

        # positional Embedding
        self.positional_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=d_model,
            name="PositionalEmbedding"
        )
        # Token Embedding
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=d_model,
            name="TokenEmbedding"
        )
        # Type Embedding
        self.type_embeddings = layers.Embedding(
            input_dim=type_size,
            output_dim=d_model,
            name="TypeEmbedding"
        )

        self.dropout = layers.Dropout(
            rate=rate,
            name="EmbedingDropout"
        )

    def call(self, inputs, training: bool = True):
        """
        inputs are in shape (batch_size, seq_len)
        the output will be in shape (batch_size, seq_len, embedding_dim=d_model)
        """
        tokens, types = inputs

        length = tf.shape(tokens)[-1]

        positions = tf.range(start=0, limit=length, delta=1)

        outputs = self.positional_embeddings(
            positions
        )  # (batch_size, seq_len, d_model)
        outputs += self.token_embeddings(tokens)
        outputs += self.type_embeddings(types)

        outputs = self.dropout(outputs, training=training)

        return outputs

    def get_config(self):
        config = super(EmbeddingLayer, self).get_config()
        config.update({"sequence_length": self.sequence_length,
                       "vocab_size": self.vocab_size,
                       "type_size": self.type_size,
                       "d_model": self.d_model,
                       "rate": self.rate})
        return config
