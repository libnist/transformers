# Import libraries
from keras import layers

# A simple pointwise feed forwar neural network
# This feedforwar nn is the vanilla neural network explained
# in the Attention is all you need paper
class PFFN(layers.Layer):
  def __init__(self, *, d_model, dense_dim, rate=.1, name="PFFN"):
    super(PFFN, self).__init__(name=name)

    self.d_model = d_model
    self.dense_dim = dense_dim
    self.rate = rate
    self.name = name

    self.dense_1 = layers.Dense(dense_dim, activation="relu")
    self.dense_2 = layers.Dense(d_model)
    self.dropout = layers.Dropout(rate)
    self.layernorm = layers.LayerNormalization()

  def call(self, inputs, training=True):

    outputs = self.dense_1(inputs) # (batch_size, seq_len, dense_dim)
    outputs = self.dense_2(outputs) # (batch_size, seq_len, d_model)
    outputs = self.dropout(outputs, training=training)
    outputs = self.layernorm(inputs + outputs)

    return outputs

  def get_config(self):
    config = {"d_model": self.d_model,
              "dense_dim": self.dense_dim,
              "rate": self.rate,
              "name": self.name}
    return config