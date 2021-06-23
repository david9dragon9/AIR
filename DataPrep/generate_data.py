import numpy as np
import tensorflow as tf
from .rasterize import rasterize
from .raw_features import features_description, state_features

def v2_build_tf_example(parsed, images=None):
  """
  Parameters:
    parsed: the raw data
    embeddings: a dictionary from models to embeddings for all egos (e.g. (# of egos, 2048))
  
  Returns:
    example: tf.train.Example
  """
  tracks = parsed['state/tracks_to_predict'] # (128,)
  tracks_before_mask = tf.squeeze(tf.where(tracks == 1), axis = 1).numpy() # indices before masking
  example = tf.train.Example()
  feature = example.features.feature
  for i, index in enumerate(tracks_before_mask):
    if images is not None:
      img = images[i]
      image_data = tf.image.encode_jpeg(img, format='rgb', quality=100)
      feature[f"image{i}/encoded"].bytes_list.value[:] = [tf.compat.as_bytes(image_data.numpy())]
  
  feature['scenario/id'].bytes_list.value[:] = [tf.compat.as_bytes(parsed['scenario/id'][0].numpy())]

  for feature_name, f in state_features.items():
    gathered = tf.gather(parsed[feature_name], tracks_before_mask).numpy()
    if f.dtype is tf.float32:
      feature[feature_name].float_list.value[:] = list(gathered.reshape(-1))
    elif f.dtype is tf.int64:
      feature[feature_name].int64_list.value[:] = list(gathered.reshape(-1))
    elif f.dtype is tf.string:
      feature[feature_name].bytes_list.value[:] = [tf.compat.as_bytes(gathered)]
  return example

def v2_process_one_raw_training_file(input_filename, output_filename):
  dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
  num_examples = 0
  for data in dataset.as_numpy_iterator():
    num_examples += 1
  print('num_examples', num_examples)
  i = 0
  all_examples = []
  for data in dataset.as_numpy_iterator():
    parsed = tf.io.parse_single_example(data, features_description)
    batch_images = rasterize(parsed)
    example = v2_build_tf_example(parsed, images = batch_images)
    all_examples.append(example)
    i += 1
  print('finished', i)
  with tf.io.TFRecordWriter(output_filename) as writer:
    for example in all_examples:
      writer.write(example.SerializeToString())

def build_tf_examples(parsed, embeddings, images=None):
  """
  Parameters:
    parsed: the raw data
    embeddings: a dictionary from models to embeddings for all egos (e.g. (# of egos, 2048))
  
  Returns:
    examples: list of tf.Examples
  """
  examples = []
  ego_x = tf.boolean_mask(parsed['state/current/x'], parsed['state/tracks_to_predict'] > 0).numpy().reshape(-1)

  for i in range(len(ego_x)):
    example = tf.train.Example()
    feature = example.features.feature
    if images is not None:
      img = images[i]
      image_data = tf.image.encode_jpeg(img, format='rgb', quality=100)
      feature['image/encoded'].bytes_list.value[:] = [tf.compat.as_bytes(image_data.numpy())]
    feature['ego_index'].int64_list.value[:] = [i]
    feature['scenario/id'].bytes_list.value[:] = [tf.compat.as_bytes(parsed['scenario/id'][0].numpy())]
    
    if len(embeddings) > 0:
      for model_name, embedding in embeddings.items():
        feature[model_name].float_list.value[:] = list(np.array(embedding[i]).reshape(-1))
    for feature_name, f in state_features.items():
      if f.dtype is tf.float32:
        feature[feature_name].float_list.value[:] = list(parsed[feature_name][i:i+1].numpy().reshape(-1))
      elif f.dtype is tf.int64:
        feature[feature_name].int64_list.value[:] = list(parsed[feature_name][i:i+1].numpy().reshape(-1))
      elif f.dtype is tf.string:
        feature[feature_name].bytes_list.value[:] = [tf.compat.as_bytes(parsed[feature_name][i:i+1].numpy())]

    examples.append(example)
  return examples

def generate_training_examples(parsed, cnn_models, state_features = state_features, include_images = False):
  """
  Parameters:
    parsed: one parsed example
    state_features: dictionary of different features
  
  Returns:
    examples: list of tf.Examples(for training)
  """
  batch_images = rasterize(parsed)
  embeddings = {}
  return build_tf_examples(parsed, embeddings, batch_images if include_images else None)

def process_one_raw_training_file(input_filename, output_filename, include_images = False):
  cnn_models = {}
  dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
  num_examples = 0
  for data in dataset.as_numpy_iterator():
    num_examples += 1
  print('num_examples', num_examples)
  i = 0
  all_examples = []
  for data in dataset.as_numpy_iterator():
    parsed = tf.io.parse_single_example(data, features_description)
    examples = generate_training_examples(parsed, cnn_models, state_features, include_images)
    all_examples += examples
    i += 1
  print('finished', i)
  with tf.io.TFRecordWriter(output_filename) as writer:
    for example in all_examples:
      writer.write(example.SerializeToString())