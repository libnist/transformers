# libraries
from keras import layers

from .masters.transformer import Master


# Vanilla transformer arcitecture for summarizatoin
class VanillaTransformer(Master):

    def __init__(self, *, encoder, decoder, output_size):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.final_layer = layers.Dense(output_size)

    def call(self, inputs, training):

        inp, tar = self.unpack_inputs(inputs, call=True)

        padding_mask, look_ahead_mask = self.create_masks(inp[0], tar[0])

        enc_output = self.encoder(inp, training, padding_mask)

        dec_output = self.decoder(tar, enc_output, training,
                                padding_mask, look_ahead_mask)
        
        final_output = self.final_layer(dec_output)

        return final_output

class FnetTransformer(Master):
    pass

class VanillaCherryPickTransformer(Master):
    pass

class FnetCherryPickTransformer(Master):
    pass
