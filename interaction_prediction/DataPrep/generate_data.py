import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from .rasterize import rasterize
from .raw_features import features_description, state_features
from .rerasterize import rerasterize_interaction as rerasterize

def build_tf_example(parsed, images=None, trajectories = None, confidences = None):
  """
  Parameters:
    parsed: the raw data
    embeddings: a dictionary from models to embeddings for all egos (e.g. (# of egos, 2048))
  
  Returns:
    example: tf.train.Example
  """
  assert images is not None, 'IMAGES IS NONE'
  interaction_tracks = parsed['state/objects_of_interest'] # (128,)
  interact_tracks_before_mask = tf.squeeze(tf.where(interaction_tracks == 1), axis = 1).numpy() # indices before masking
  example = tf.train.Example()
  feature = example.features.feature
  for i, index in enumerate(interact_tracks_before_mask):
    if images is not None:
      img = images[i]
      image_data = tf.image.encode_jpeg(img, format='rgb', quality=100)
      feature[f"image{i}/encoded"].bytes_list.value[:] = [tf.compat.as_bytes(image_data.numpy())]
  
  feature['scenario/id'].bytes_list.value[:] = [tf.compat.as_bytes(parsed['scenario/id'][0].numpy())]

  if trajectories is not None:
    feature['trajectories'].float_list.value[:] = list(trajectories.numpy().reshape(-1))
  if confidences is not None:
    feature['confidences'].float_list.value[:] = list(confidences.numpy().reshape(-1))

  for feature_name, f in state_features.items():
    gathered = tf.gather(parsed[feature_name], interact_tracks_before_mask).numpy()
    if f.dtype is tf.float32:
      feature[feature_name].float_list.value[:] = list(gathered.reshape(-1))
    elif f.dtype is tf.int64:
      feature[feature_name].int64_list.value[:] = list(gathered.reshape(-1))
    elif f.dtype is tf.string:
      feature[feature_name].bytes_list.value[:] = [tf.compat.as_bytes(gathered)]
  return example

def process_one_raw_training_file(input_filename, output_filename):
  dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
  num_examples = 0
  for data in dataset.as_numpy_iterator():
    num_examples += 1
  print('num_examples', num_examples)
  i = 0
  all_examples = []
  for data in dataset.as_numpy_iterator():
    parsed = tf.io.parse_single_example(data, features_description)
    if len(tf.where(parsed['state/objects_of_interest'] == 1).numpy()) != 2:
      continue
    batch_images = rasterize(parsed)
    example = build_tf_example(parsed, images = batch_images)

    all_examples.append(example)
    i += 1
  print('finished', i)
  with tf.io.TFRecordWriter(output_filename) as writer:
    for example in all_examples:
      writer.write(example.SerializeToString())

def process_one_raw_training_file_for_rr(input_filename, output_filename, output_filename_rr, model):
  dataset = tf.data.TFRecordDataset(input_filename, compression_type='')
  num_examples = 0
  for data in dataset.as_numpy_iterator():
    num_examples += 1
  print('num_examples', num_examples)
  i = 0
  all_examples = []
  all_examples_rr = []
  for data in dataset.as_numpy_iterator():
    parsed = tf.io.parse_single_example(data, features_description)
    if len(tf.where(parsed['state/objects_of_interest'] == 1).numpy()) != 2:
      continue
    batch_images = rasterize(parsed)
    example = build_tf_example(parsed, images = batch_images)
    all_examples.append(example)
    i += 1
    batch_images, pred_trajectory, confidences = rerasterize(model, example)
    rr_example = build_tf_example(parsed, images = batch_images, trajectories = pred_trajectory, confidences = confidences)
    all_examples_rr.append(rr_example)
  print('finished', i)
  if output_filename:
    with tf.io.TFRecordWriter(output_filename) as writer:
      for example in all_examples:
        writer.write(example.SerializeToString())
  with tf.io.TFRecordWriter(output_filename_rr) as writer:
    for example in all_examples_rr:
      writer.write(example.SerializeToString())
