# Import libraries
import tensorflow as tf


class AIAYNSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Used to create the learning rates based on the proposed method in the
    Attention is all you need paper. you only need to pass dimention of your
    model as `d_model`.
    """

    def __init__(
        self, d_model: int, warmup_steps: int = 4000
    ) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        super(AIAYNSchedule, self).__init__()

        self.d_model_to_serialize = d_model
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {"d_model": self.d_model_to_serialize,
                  "warmup_steps": self.warmup_steps}
        return config


def aiayn_adam(learning_rate: AIAYNSchedule) -> tf.keras.optimizers.Adam:
    """
    This function accepts a learning rate schedule that is based on 
    Attention is all you need paper, and creates the Adam optimizer that is
    used in the mentioned paper.
    """
    return tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )
