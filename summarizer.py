# Import libraries
import tensorflow as tf


class Summarizer(tf.Module):

    def __init__(self, tokenizers, transformer):

        self.tokenizers = tokenizers
        self.transformer = transformer

    def __call__(self, document, max_length=64):

        assert isinstance(document, tf.Tensor), "Document is not a Tensor."

        if len(document.shape) == 0:
            document = document[tf.newaxis]

        document = self.tokenizers.doc.tokenize(document)
        document = (document[0].to_tensor(), document[1].to_tensor())

        encoder_input = document

        start_dot_sep_end = self.tokenizers.summ.tokenize(['. A'])[0][0]
        start = start_dot_sep_end[0][tf.newaxis]
        end = start_dot_sep_end[-1][tf.newaxis]
        sep = start_dot_sep_end[-2][tf.newaxis]

        output_array = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True
        )
        output_array = output_array.write(0, start)

        type_ = [0]
        output_type = tf.TensorArray(
            dtype=tf.int64, size=0, dynamic_size=True
        )
        output_type = output_type.write(0, type_)

        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            types = tf.transpose(output_type.stack())
            output = (output, types)
            predictions = self.transformer(
                [encoder_input, output], training=False
            )

            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

            predicted_id = tf.argmax(predictions, axis=-1)

            if predicted_id[0] == sep:
                type_[0] += 1

            output_array = output_array.write(i+1, predicted_id[0])
            output_type = output_type.write(i+1, type_)

            if predicted_id == end:
                break

        output = tf.transpose(output_array.stack())

        text = self.tokenizers.doc.detokenize(output)[0]

        tokens = self.tokenizers.sum.lookup(output)[0]

        return text, tokens
