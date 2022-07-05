# Import libs
import tensorflow as tf
import tensorflow_text as text
import pathlib
import re
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

__all__ = ["make_token_from_dataset",
           "tokenizers_from_vocab_path"]


reserved_tokens = ["[pad]", "[unk]", "[start]", "[end]", "[sep]"]


def preprocess(inputs):
    pattern = r".*\(CNN\)\s+--|^\(CNN\)|.*(UPDATED:\.|Last updated|PUBLISHED:\.).*\d{4}\.|^By\..{5,50}\.|[^\x00-\x7F]"
    inputs = tf.strings.regex_replace(inputs, pattern, "")
    inputs = tf.strings.strip(inputs)
    inputs = tf.strings.regex_replace(inputs, r"\n|[^\x00-\x7F]", " ")
    return inputs


def write_vocab_file(filepath, vocab):
    with open(filepath, "w") as f:
        for token in vocab:
            print(token, file=f)


def read_vocab(path):
    with open(path, "r") as f:
        vocab = f.readlines()
    vocab = list(map(lambda token: token.replace("\n", ""), vocab))
    return vocab


def make_token_from_dataset(vocab_size, data, file_path):

    preprocessed_data = data.map(lambda x: preprocess(x))
    bert_tokenizer_params = dict(lower_case=True)

    bert_vocab_args = dict(
        vocab_size=vocab_size,
        reserved_tokens=reserved_tokens,
        bert_tokenizer_params=bert_tokenizer_params,
        learn_params={}
    )

    vocab = bert_vocab.bert_vocab_from_dataset(
        preprocessed_data.batch(1000).prefetch(tf.data.AUTOTUNE),
        **bert_vocab_args
    )

    write_vocab_file(file_path, vocab)

    return vocab


def get_token_of(tokens, sign):
    return [tf.argmax(tf.constant(tokens) == sign)]


start = get_token_of(reserved_tokens, "[start]")
end = get_token_of(reserved_tokens, "[end]")
sep = get_token_of(reserved_tokens, "[sep]")


@tf.function
def custom_tokenize(strings, tokenizer):
    documents = tf.strings.regex_replace(strings,
                                         r"((\!|\?)|(\.)) ([A-Z])",
                                         r"\1 [break] \2")
    documents = tf.strings.split(documents, "[break]")
    documents = tokenizer.tokenize(documents).merge_dims(-2, -1)

    number_of_docs = documents.bounding_shape()[0]
    doc_sizes = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)

    whole_docs = tf.ragged.constant([[]], dtype=tf.int64)

    one = tf.constant(1, dtype=tf.int64)

    for i in tf.range(number_of_docs, dtype=tf.int32):
        n_sents = documents[i].bounding_shape()[0] - one

        seps = tf.expand_dims(tf.repeat(sep, n_sents), axis=1)
        seps = tf.concat(
            [seps, tf.ragged.constant([[]], dtype=tf.int64)], axis=0)

        doc = tf.concat([documents[i], seps], axis=1).merge_dims(-2, -1)
        doc = tf.RaggedTensor.from_tensor([doc])
        doc_sizes = doc_sizes.write(i, doc.bounding_shape()[1])

        whole_docs = tf.concat([whole_docs, doc], axis=1)

    documents = tf.RaggedTensor.from_row_lengths(whole_docs[0],
                                                 doc_sizes.stack())

    starts = tf.expand_dims(tf.repeat(start, number_of_docs), axis=1)
    ends = tf.expand_dims(tf.repeat(end, number_of_docs), axis=1)

    documents = tf.concat([starts, documents, ends], axis=1)

    return documents


@tf.function
def get_ids(tokens):

    number_of_tokens = tokens.bounding_shape()[0]

    ids = tf.ragged.constant([[]], dtype=tf.int64)
    sizes = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    zero = tf.constant([[0]], dtype=tf.int64)
    for i in tf.range(number_of_tokens, dtype=tf.int32):
        sizes = sizes.write(i, tf.shape(tokens[i])[0])
        sep_indexs = tf.where((tokens[i] == sep) | (tokens[i] == end)) + 1
        shifted_indexs = tf.concat([zero, sep_indexs[:-1]], axis=0)
        sep_sizes = sep_indexs - shifted_indexs
        for j in tf.range(tf.shape(sep_sizes)[0], dtype=tf.int64):
            temp = tf.expand_dims(tf.repeat(j, sep_sizes[j]), axis=0)
            ids = tf.concat([ids, temp], axis=1)
    ids = tf.RaggedTensor.from_row_lengths(ids[0], sizes.stack())
    return ids


def cleanup_text(reserved_tokens, token_txt):
    # Drop the reserved tokens, except for "[UNK]".
    bad_tokens = [re.escape(tok) for tok in reserved_tokens if tok != "[UNK]"]
    bad_token_re = "|".join(bad_tokens)

    bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
    result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

    # Join them into strings.
    result = tf.strings.reduce_join(result, separator=' ', axis=-1)

    return result


class CustomTokenizer(tf.Module):
    def __init__(self, reserved_tokens, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        self._reserved_tokens = reserved_tokens
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        vocab = pathlib.Path(vocab_path).read_text().splitlines()
        self.vocab = tf.Variable(vocab)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        # These `get_*` methods take no arguments
        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        strings = preprocess(strings)
        tokens = custom_tokenize(strings, self.tokenizer)
        tokens_ids = get_ids(tokens)
        return tokens, tokens_ids

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return cleanup_text(self._reserved_tokens, words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)


def tokenizers_from_vocab_path(doc_vocab_path, sum_vocab_path):
    tokenizers = tf.Module()
    tokenizers.doc = CustomTokenizer(reserved_tokens, doc_vocab_path)
    tokenizers.sum = CustomTokenizer(reserved_tokens, sum_vocab_path)
    return tokenizers
