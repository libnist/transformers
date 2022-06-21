# libraries
import tensorflow as tf
import keras

from keras import layers

from .masters.transformer import Master

# Vanilla transformer arcitecture for summarizatoin
class VanillaTransformer(Master):
    pass

class FnetTransformer(Master):
    pass

class VanillaCherryPickTransformer(Master):
    pass

class FnetCherryPickTransformer(Master):
    pass
