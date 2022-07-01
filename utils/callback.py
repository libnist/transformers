# Import libraries
import tensorflow as tf


class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, ckpt_path, save_per_epoch=5):
        super(SaveCallback, self).__init__()

        self.save_per_epoch = save_per_epoch
        self.ckpt_path = ckpt_path

    def on_train_begin(self, logs=None):
        self.model.checkpoint(self.ckpt_path)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_per_epoch == 0:
            self.model.save(epoch)
