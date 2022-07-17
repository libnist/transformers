# Import libraries
import tensorflow as tf

# __all__ defines what can be accessed from outside
__all__ = ["SaveCallback"]


class SaveCallback(tf.keras.callbacks.Callback):
    """
    This callback saves the whole model every `save_per_epoch` in ckpt_path.
    Only last 5 checkpoints are being kept. If a model uses this callback, the 
    last checkpoint saved can be restored by `Model.checkpoint(ckpt_path)`.
    Also by giving the same `ckpt_path` when fitting the model, fitting will be
    continued from the last checkpoint in `ckpt_path`.
    """
    def __init__(self, ckpt_path: str, save_per_epoch: int = 5):
        super(SaveCallback, self).__init__()

        self.save_per_epoch = save_per_epoch
        self.ckpt_path = ckpt_path

    def on_train_begin(self, logs=None):
        self.model.checkpoint(self.ckpt_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_per_epoch == 0:
            self.model.save(epoch)
