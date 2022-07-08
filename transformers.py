# Import libraries
from keras import layers

from .masters.transformer import Master


# Vanilla transformer arcitecture for summarizatoin
class AIAYNTransformer(Master):
    """
    The overall architecture of this transformer will be the same as the one
    introduced in Attention is all you need paper, meaning that there nothing
    extra or less in the encoder or decoder layesr. althogh attention
    mechanisms are the same as their encoder layers so they can be different, 
    for example an encoder with fnet attention could be built with this 
    transformer as long as it follows the same input output structure of the
    transformer propsed in AIAYN paper.
    """
    def __init__(self, *, encoder, decoder, output_size):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        self.final_layer = layers.Dense(units=output_size)

    def call(self, inputs, training):

        inp, tar = self.unpack_inputs(inputs=inputs, call=True)

        padding_mask, look_ahead_mask = self.create_masks(inp=inp[0],
                                                          tar=tar[0])

        enc_outputs = self.encoder(
            inputs=inp, training=training, padding_mask=padding_mask
        )

        dec_outputs = self.decoder(
            inputs=tar, enc_outputs=enc_outputs, training=training,
            padding_mask=padding_mask, look_ahead_mask=look_ahead_mask
        )

        final_output = self.final_layer(dec_outputs)

        return final_output


class FnetTransformer(Master):
    pass


class VanillaCherryPickTransformer(Master):
    pass


class FnetCherryPickTransformer(Master):
    pass
