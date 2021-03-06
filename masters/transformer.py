# Import libraries
import tensorflow as tf
import keras

from ..utils.masks import *


# In this module a transformer will be implemented. this transformer only
# contains the training_step of a transformer, so we can make custom
# transformers based on this MasterTransformer and just define the architecture
# and call() function. for the sake of different type of inputs, an
# unpack_inputs() function will be declared that could be overrode in neccesary
# cases for further custumizatoins. It also contains the masking tools in it so
#  we don't have to import it for many times.


class Master(keras.Model):
    """
    To be inherited by transformers with different approaches and layers.
    Can be furthur customized by ovreriding the methods below. customize
    `unpack_inputs` for different type of inputs. 
    """

    def train_step(self, inputs):
        """
        The things that will happen in each train_step.
        """
        data = self.unpack_inputs(inputs=inputs, call=False)

        with tf.GradientTape() as tape:
            predictions = self((data["inputs"],
                                data["inp_targets"]),
                               training=True)
            loss = self.compiled_loss(y_true=data["real_targets"],
                                      y_pred=predictions,
                                      regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(data["real_targets"], predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, inputs):
        """
        What will happen in each test step.
        """
        data = self.unpack_inputs(inputs=inputs, call=False)

        # Compute predictions
        y_pred = self((data["inputs"], data["inp_targets"]), training=False)

        # Updates the metrics tracking the loss
        self.compiled_loss(data["real_targets"],
                           y_pred,
                           regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(data["real_targets"], y_pred)

        return {m.name: m.result() for m in self.metrics}

    def unpack_inputs(self, inputs, call=True):
        """
        This functoin is designed to prepare the inputs for the
        transformer, in case of training or interpreting.
        In case of overriding consider both cases.
        In the vanilla case of this functoin we expect inputs to have
        types that defines separate sentences in a document.
        If you don't want this you can just override this function
        and apply it to you call() or your own train_step().
        """
        tar, tar_types = inputs[1]
        if not call:
            inp_targets = tar[:, :-1], tar_types[:, :-1]
            real_targets = tar[:, 1:]
            outputs = {"inputs": inputs[0],
                       "inp_targets": inp_targets,
                       "real_targets": real_targets}
        else:
            outputs = inputs
        return outputs

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

    def checkpoint(self, ckpt_path):
        """
        creates checkpoint and checkpoint manager to save or load models.
        if the checkpoint exists the last state will be restored.
        """
        if not self.optimizer:
            raise ValueError("Optimizer is not compiled.")

        self.ckpt = tf.train.Checkpoint(transformer=self,
                                        optimizer=self.optimizer)

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                       ckpt_path,
                                                       max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!")

    def save(self, epoch=None):
        """
        if a checkpoint manager exists the model will be saved, otherwise 
        call the `checkpoint` method first.
        """
        if not self.ckpt_manager:
            raise ValueError("ckpt_manager is not defined")
        self.ckpt_manager.save()

    def get_config(self):
        config = super().get_config()
        config.update({"ckpt_path": self.ckpt_path})
        return config
