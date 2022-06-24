from keras import layers

from .attention import FnetAttention
from .feedforward import PFFN

enc_access = ["VanillaEncoderLayer",
              "FnetEncoderLayer"]

# Vanilla EncoderLayer
class VanillaEncoderLayer(layers.Layer):
  def __init__(self, *, d_model, num_heads,
               dense_dim=1024, rate=.1,
               name="VanillaEncoderLayer"):
    super(VanillaEncoderLayer, self).__init__(name=name)

    self.d_model = d_model
    self.num_heads = num_heads
    self.dense_dim = dense_dim
    self.rate = rate
    self.name = name

    self.mha = layers.MultiHeadAttention(num_heads=num_heads,
                                         key_dim=d_model,
                                         dropout=rate)
    self.pffn = PFFN(d_model=d_model,
                     dense_dim=dense_dim,
                     rate=rate)

  def call(self, inputs, training=True, padding_mask=None):

    outputs = self.mha(query=inputs, key=inputs, value=inputs,
                       attention_mask=padding_mask, training=training)
    outputs = self.pffn(outputs, training)

    return outputs

def get_config(self):
    config = {"d_model": self.d_model,
              "num_heads": self.num_heads,
              "dense_dim": self.dense_dim,
              "rate": self.rate,
              "name": self.name}
    return config

# Fnet-EncoderLayer
class FnetEncoderLayer(layers.Layer):
  def __init__(self, *, d_model, dense_dim=1024,
               with_dense=False, rate=.1,
               name="FnetEncoderLayer"):
    super(FnetEncoderLayer, self).__init__(name=name)

    self.d_model = d_model
    self.dense_dim = dense_dim,
    self.with_dense = with_dense
    self.rate = rate
    self.name = name

    self.fnet = FnetAttention(with_dense=with_dense,
                              d_model=d_model,
                              rate=rate)
    self.pffn = PFFN(d_model=d_model,
                     dense_dim=dense_dim,
                     rate=rate)

  def call(self, inputs, training=True, **kwargs):

    outputs = self.fnet(inputs, training)
    outputs = self.pffn(outputs, training)

    return outputs

def get_config(self):
    config = {"d_model": self.d_model,
              "dense_dim": self.dense_dim,
              "with_dense": self.with_dense,
              "rate": self.rate,
              "name": self.name}
    return config