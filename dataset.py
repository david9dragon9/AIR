import tensorflow as tf
import numpy as np

# Features of other agents.
e_scenario_features = {
    'scenario/id':
        tf.io.FixedLenFeature([1], tf.string, default_value = "")
}
e_state_features = {
    'state/id':
        tf.io.FixedLenFeature([1], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([1], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([1], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([1], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([1, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([1, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([1, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([1, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([1, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([1, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([1, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([1, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([1, 10], tf.float32, default_value=None),
}

e_features = {
    'ego_index':
        tf.io.FixedLenFeature([1,], tf.int64, default_value=None),
    'image/encoded':
        tf.io.VarLenFeature(dtype=tf.string)
}

e_features_description = {}
e_features_description.update(e_scenario_features)
e_features_description.update(e_state_features)
e_features_description.update(e_features)

def _parse_common(decoded_example):
  scenario_id = decoded_example['scenario/id'] # [1]
  object_id = decoded_example['state/id'] # [1]

  x = decoded_example['state/current/x']
  y = decoded_example['state/current/y']
  yaw = decoded_example['state/current/bbox_yaw']

  past_states = tf.stack([
      decoded_example['state/past/x'],
      decoded_example['state/past/y'],
      decoded_example['state/past/length'],
      decoded_example['state/past/width'],
      decoded_example['state/past/bbox_yaw'],
      decoded_example['state/past/velocity_x'],
      decoded_example['state/past/velocity_y']
  ], -1)

  cur_states = tf.stack([
      decoded_example['state/current/x'],
      decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1)

  input_states = tf.concat([past_states, cur_states], 1)[..., :7]

  future_states = tf.stack([
      decoded_example['state/future/x'],
      decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1)
  past_is_valid = decoded_example['state/past/valid'] > 0
  current_is_valid = decoded_example['state/current/valid'] > 0
  future_is_valid = decoded_example['state/future/valid'] > 0
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1)
  
  inputs = {
            'is_sdc': decoded_example['state/is_sdc'], # (1,)
            'gt_future_states': gt_future_states, # (1, 91, 7)
            'gt_future_is_valid': gt_future_is_valid, # (1, 91)
            'past_states':past_states, # (1, 10, 7)
            'object_type': decoded_example['state/type'], # (1, )
            'x':x, # (1,)
            'y':y, # (1, )
            'yaw':yaw, # (1, )
            'scenario_id':scenario_id,
            'object_id': object_id}
  return inputs


def _parse(value):
  decoded_example = tf.io.parse_single_example(value, e_features_description)
  inputs = _parse_common(decoded_example)
  encoded=tf.sparse.to_dense(decoded_example['image/encoded'])[0]
  inputs['image'] = tf.image.decode_jpeg(encoded)
  return inputs

def _parse_without_image(value):
  decoded_example = tf.io.parse_single_example(value, e_features_description)
  return _parse_common(decoded_example)

def is_valid(x):
  avails_float = tf.cast(x["gt_future_is_valid"], tf.float32) # (1, 91)
  dist = (x["gt_future_states"][0, 12:, 0:2] - x["gt_future_states"][0, 11:90, 0:2])**2 # (79, 2)
  dist = tf.reduce_sum(dist, axis=1) # (79,)
  dist *= avails_float[0, 11:90]*avails_float[0, 12:] # (79,)
  dist = tf.reduce_max(dist, axis=0) # ()
  return dist < 36

def get_dataset(file_pattern, batch_size=32, filter_valid=False):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4), cycle_length=8)\
  .map(_parse, num_parallel_calls=8)
  if filter_valid:
    dataset = dataset.filter(lambda x: is_valid(x))
  return dataset.batch(batch_size)

def _cyc_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return example['state/type'][0] == 3.

def get_cyclist_dataset(file_pattern, batch_size=32, filter_valid=False):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _cyc_only(y)), cycle_length=8)\
  .map(_parse, num_parallel_calls=8)
  if filter_valid:
      dataset = dataset.filter(lambda x: is_valid(x))
  return dataset.batch(batch_size)

def get_eval_dataset(data_type, eval_file_pattern, batch_size=32):
  if data_type == "cyclist":
    dataset = get_cyclist_dataset(eval_file_pattern, batch_size)
  elif data_type == "ped":
    dataset = get_ped_dataset(eval_file_pattern, batch_size)
  elif data_type == "veh":
    dataset = get_veh_dataset(eval_file_pattern, batch_size)
  else:
    dataset = get_dataset(eval_file_pattern, batch_size)
  return dataset

ot_feature_desc = {
    'state/type':
        tf.io.FixedLenFeature([1], tf.float32, default_value=None),
}
def _ped_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return example['state/type'][0] == 2

def get_ped_dataset(file_pattern, batch_size=32, filter_valid=False):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _ped_only(y)), cycle_length=8)\
  .map(_parse, num_parallel_calls=8)
  if filter_valid:
      dataset = dataset.filter(lambda x: is_valid(x))
  return dataset.batch(batch_size)

def _veh_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return example['state/type'][0] == 1

def get_veh_dataset(file_pattern, batch_size=32, filter_valid=False):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _veh_only(y)), cycle_length=8)\
  .map(_parse, num_parallel_calls=8)
  if filter_valid:
    dataset = dataset.filter(lambda x: is_valid(x))
  return dataset.batch(batch_size)

def get_deterministic_dataset(file_pattern):
  file_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4), cycle_length=1)\
  .map(_parse).batch(32)
  return dataset

def get_dataset_for_clustering(file_pattern):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4), cycle_length=8)\
  .map(_parse_without_image, num_parallel_calls=8).batch(32)
  return dataset

def _object_type_only(data, object_type):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return example['state/type'][0] == object_type 

def get_extended_dataset(train_file_pattern, validation_file_pattern, object_type):
  train_file_dataset = tf.data.Dataset.list_files(train_file_pattern)
  validation_file_dataset = tf.data.Dataset.list_files(validation_file_pattern)
  combined_dataset = train_file_dataset.concatenate(validation_file_dataset).shuffle(1100)
  dataset = combined_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _object_type_only(y, object_type)), cycle_length=8)\
  .map(_parse, num_parallel_calls=8)
  return dataset

def combine_datasets(veh_dataset, ped_dataset, cyc_dataset, veh_val, ped_val, cyc_val):
  veh_dataset = veh_dataset.repeat()
  ped_dataset = ped_dataset.repeat()
  cyc_dataset = cyc_dataset.repeat()
  dataset = tf.data.experimental.sample_from_datasets([veh_dataset, ped_dataset, cyc_dataset], [veh_val, ped_val, cyc_val])
  return dataset

def get_weighted_dataset(train_file_pattern, validation_file_pattern, veh_val, ped_val, cyc_val, batch_size=32):
  veh_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=1)
  ped_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=2)
  cyc_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=3)
  return combine_datasets(veh_dataset, ped_dataset, cyc_dataset, veh_val, ped_val, cyc_val).batch(batch_size)