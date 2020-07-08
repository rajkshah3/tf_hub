import pandas as pd
import numpy as np
import tensorflow as tf 
import tensorflow_hub as hub
import tokenization
from tensorflow.keras import backend as K

tb = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)
# Initialize session
sess = tf.Session()
bert_path = 'https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1'
dataset_size = 500

def make_rating_binary(data):

    if(data>5):
        data = 1
    else:
        data=0

    return data

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

class BertLayer(tf.keras.layers.Layer):
    def __init__(self, n_fine_tune_layers=2,trainable=True, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = trainable
        self.output_size = 300
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )

        trainable_vars = self.bert.variables
        # Remove unused layers
        tuning_layers = [ 'layer_{}'.format(x) for x in range(10-self.n_fine_tune_layers,10)]

        # trainable_vars = [var for var in trainable_vars if not [tuning_layers] in var.name]
        trainable_vars = [var for var in trainable_vars if any(x in var.name for x in tuning_layers)]
        # Select how many layers to fine tune
        # trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        # import pdb; pdb.set_trace()  # breakpoint 8010aa9d //
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
            "sequence_output"
        ]
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in examples:
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples

def create_tokenizer_from_hub_module(bert_hub_module_handle):
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(bert_hub_module_handle)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
  return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

def build_model(max_seq_length=100,classifier='bert_cls_input'):
    # Build model
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    # Instantiate the custom Bert Layer defined above
    bert_output = BertLayer(n_fine_tune_layers=2)(bert_inputs)
    # Build the rest of the classifier

    if(classifier == 'flatten'):
        bert_output = flatten(bert_output)
    if(classifier == 'lstm'):
        bert_output = lstm(bert_output)
    if(classifier == 'bert_cls_input'):
        bert_output = bert_cls_input(bert_output)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=bert_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def flatten(bert_output):
    bert_output = tf.keras.layers.Reshape(target_shape=(max_seq_length,np.shape(bert_output)[2]))(bert_output)
    bert_output = tf.keras.layers.Flatten()(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    return bert_output

def lstm(bert_output):
    bert_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    return bert_output

def lstm_sequence(bert_output):
    bert_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,return_sequences=True))(bert_output)
    bert_output = tf.keras.layers.Reshape(target_shape=(max_seq_length,np.shape(bert_output)[2]))(bert_output)
    bert_output = tf.keras.layers.Flatten()(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    return bert_output

def time_distributed_min(bert_output):
    bert_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64,activation='relu'))(bert_output)
    bert_output = tf.keras.layers.Reshape(target_shape=(max_seq_length,np.shape(bert_output)[2]))(bert_output)
    bert_output = tf.keras.layers.Flatten()(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    return bert_output


def bert_cls_input(bert_output):

    import pdb; pdb.set_trace()  # breakpoint d609fe71 //
    bert_output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256))(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    bert_output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)
    return bert_output

df = pd.read_csv('~/data/movie_review_data/review_df.csv')
df = df.sample(frac=1,random_state=1234)
df['rating'] = df['rating'].apply(make_rating_binary)

train_df, test_df = df.head(dataset_size), df.tail(dataset_size)

max_seq_length  = 50
# Create datasets (Only take up to `max_seq_length` words for memory)
train_text = train_df['review_text'].tolist()
train_text = [' '.join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df['rating'].tolist()

test_text = test_df['review_text'].tolist()
test_text = [' '.join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df['rating'].tolist()


tokenization = create_tokenizer_from_hub_module(bert_path)

train = convert_text_to_examples(train_text,train_label)
train_input = convert_examples_to_features(tokenization,train,max_seq_length=max_seq_length)

test = convert_text_to_examples(test_text,test_label)
test_input = convert_examples_to_features(tokenization,test,max_seq_length=max_seq_length)


model = build_model(max_seq_length=max_seq_length,classifier='bert_cls_input')
initialize_vars(sess)
import pdb; pdb.set_trace()  # breakpoint 08328f53 //

prediction = model.predict(train_input)
# fake_data = np.ones(shape=(dataset_size,max_seq_length,768))
model.fit(train_input,train_label,batch_size=100,epochs=2,callbacks=[tb])
import pdb; pdb.set_trace()  # breakpoint 87a2b9fd //

print('done')