import tensorflow as tf
import numpy as np

i_scenario_features = {
    'scenario/id':
        tf.io.FixedLenFeature([1], tf.string, default_value = None)
}

i_state_features = {
    'state/id':
        tf.io.FixedLenFeature([2], tf.float32, default_value=None),
    'state/type':
        tf.io.FixedLenFeature([2], tf.float32, default_value=None),
    'state/is_sdc':
        tf.io.FixedLenFeature([2], tf.int64, default_value=None),
    'state/tracks_to_predict':
        tf.io.FixedLenFeature([2], tf.int64, default_value=None),
    'state/current/bbox_yaw':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/height':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/length':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/timestamp_micros':
        tf.io.FixedLenFeature([2, 1], tf.int64, default_value=None),
    'state/current/valid':
        tf.io.FixedLenFeature([2, 1], tf.int64, default_value=None),
    'state/current/vel_yaw':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/velocity_x':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/velocity_y':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/width':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/x':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/y':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/current/z':
        tf.io.FixedLenFeature([2, 1], tf.float32, default_value=None),
    'state/future/bbox_yaw':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/height':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/length':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/timestamp_micros':
        tf.io.FixedLenFeature([2, 80], tf.int64, default_value=None),
    'state/future/valid':
        tf.io.FixedLenFeature([2, 80], tf.int64, default_value=None),
    'state/future/vel_yaw':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/velocity_x':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/velocity_y':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/width':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/x':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/y':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/future/z':
        tf.io.FixedLenFeature([2, 80], tf.float32, default_value=None),
    'state/past/bbox_yaw':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/height':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/length':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/timestamp_micros':
        tf.io.FixedLenFeature([2, 10], tf.int64, default_value=None),
    'state/past/valid':
        tf.io.FixedLenFeature([2, 10], tf.int64, default_value=None),
    'state/past/vel_yaw':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/velocity_x':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/velocity_y':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/width':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/x':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/y':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
    'state/past/z':
        tf.io.FixedLenFeature([2, 10], tf.float32, default_value=None),
}

i_features = {
    'image0/encoded':
        tf.io.FixedLenFeature([1], tf.string, default_value=None),
    'image1/encoded':
        tf.io.FixedLenFeature([1], tf.string, default_value=None),
}

i_features_description = {}
i_features_description.update(i_scenario_features)
i_features_description.update(i_state_features)
i_features_description.update(i_features)

def _parse_no_swap(value):
  decoded_example = tf.io.parse_single_example(value, i_features_description)
  return parse_example_no_swap(decoded_example)

def parse_example_no_swap(decoded_example):
  scenario_id = decoded_example['scenario/id'] # [1]
  object_id = decoded_example['state/id'] # [2]

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
  ], -1) # (2, 10, 7)

  cur_states = tf.stack([
      decoded_example['state/current/x'],
      decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1) # (2, 1, 7)

  input_states = tf.concat([past_states, cur_states], 1)[..., :7] # (2, 11, 7)

  future_states = tf.stack([
      decoded_example['state/future/x'],
      decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1) # (2, 80, 7)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1) # (2, 91, 7)
  past_is_valid = decoded_example['state/past/valid'] > 0 # (2, 10)
  current_is_valid = decoded_example['state/current/valid'] > 0 # (2,1)
  future_is_valid = decoded_example['state/future/valid'] > 0 # (2, 80)
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1) # (2, 91)


  encoded_0 = decoded_example['image0/encoded'][0] # (224,448,3)
  encoded_1 = decoded_example['image1/encoded'][0] # (224,448,3)
  is_sdc = decoded_example['state/is_sdc']

  object_type = decoded_example['state/type']
  image_0 =  tf.image.decode_jpeg(encoded_0)
  image_1 =  tf.image.decode_jpeg(encoded_1)
  inputs = {
            'is_sdc': is_sdc,
            'gt_future_states': gt_future_states,# (2, 91, 7)
            'gt_future_is_valid': gt_future_is_valid,# (2, 91)
            'past_states': past_states,# (2, 10, 7)
            'object_type': object_type,# (2, )
            'x': x,# (2,)
            'y': y,# (2, )
            'yaw':yaw,# (2, )
            'image_0': image_0,
            'image_1': image_1,
            'scenario_id':scenario_id,
            'object_id': object_id
            }
  return inputs

def _parse(value):
  decoded_example = tf.io.parse_single_example(value, i_features_description)
  return parse_example(decoded_example)

def parse_example(decoded_example):
  scenario_id = decoded_example['scenario/id'] # [1]
  object_id = decoded_example['state/id'] # [2]

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
  ], -1) # (2, 10, 7)

  cur_states = tf.stack([
      decoded_example['state/current/x'],
      decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1) # (2, 1, 7)

  input_states = tf.concat([past_states, cur_states], 1)[..., :7] # (2, 11, 7)

  future_states = tf.stack([
      decoded_example['state/future/x'],
      decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1) # (2, 80, 7)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1) # (2, 91, 7)
  past_is_valid = decoded_example['state/past/valid'] > 0 # (2, 10)
  current_is_valid = decoded_example['state/current/valid'] > 0 # (2,1)
  future_is_valid = decoded_example['state/future/valid'] > 0 # (2, 80)
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1) # (2, 91)


  encoded_0 = decoded_example['image0/encoded'][0] # (224,448,3)
  encoded_1 = decoded_example['image1/encoded'][0] # (224,448,3)
  is_sdc = decoded_example['state/is_sdc']

  object_type = decoded_example['state/type']
  swap = tf.cast(object_type[0] > object_type[1], tf.int32)
  indices = swap * tf.constant([1, 0]) + (1-swap) * tf.constant([0, 1])
  orig_image_0 =  tf.image.decode_jpeg(encoded_0)
  orig_image_1 =  tf.image.decode_jpeg(encoded_1)
  swap_uint8 = tf.cast(swap, tf.uint8)
  image_0 = swap_uint8*orig_image_1 + (1-swap_uint8)*orig_image_0
  image_1 = swap_uint8*orig_image_0 + (1-swap_uint8)*orig_image_1
  inputs = {
            'is_sdc': tf.gather(is_sdc, indices, axis=0),
            'gt_future_states': tf.gather(gt_future_states, indices, axis=0),# (2, 91, 7)
            'gt_future_is_valid': tf.gather(gt_future_is_valid, indices, axis=0),# (2, 91)
            'past_states':tf.gather(past_states, indices, axis=0),# (2, 10, 7)
            'object_type': tf.gather(object_type, indices, axis=0),# (2, )
            'x':tf.gather(x, indices, axis=0),# (2,)
            'y':tf.gather(y, indices, axis=0),# (2, )
            'yaw':tf.gather(yaw, indices, axis=0),# (2, )
            'image_0': image_0,
            'image_1': image_1,
            'scenario_id':scenario_id,
            'object_id': tf.gather(object_id, indices, axis=0),
            'indices': indices
            }
  return inputs

def _parse_without_image(value):
  decoded_example = tf.io.parse_single_example(value, i_features_description)

  scenario_id = decoded_example['scenario/id'] # [1]
  object_id = decoded_example['state/id'] # [2]

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
  ], -1) # (2, 10, 7)

  cur_states = tf.stack([
      decoded_example['state/current/x'],
      decoded_example['state/current/y'],
      decoded_example['state/current/length'],
      decoded_example['state/current/width'],
      decoded_example['state/current/bbox_yaw'],
      decoded_example['state/current/velocity_x'],
      decoded_example['state/current/velocity_y']
  ], -1) # (2, 1, 7)

  input_states = tf.concat([past_states, cur_states], 1)[..., :7] # (2, 11, 7)

  future_states = tf.stack([
      decoded_example['state/future/x'],
      decoded_example['state/future/y'],
      decoded_example['state/future/length'],
      decoded_example['state/future/width'],
      decoded_example['state/future/bbox_yaw'],
      decoded_example['state/future/velocity_x'],
      decoded_example['state/future/velocity_y']
  ], -1) # (2, 80, 7)

  gt_future_states = tf.concat([past_states, cur_states, future_states], 1) # (2, 91, 7)
  past_is_valid = decoded_example['state/past/valid'] > 0 # (2, 10)
  current_is_valid = decoded_example['state/current/valid'] > 0 # (2,1)
  future_is_valid = decoded_example['state/future/valid'] > 0 # (2, 80)
  gt_future_is_valid = tf.concat(
      [past_is_valid, current_is_valid, future_is_valid], 1) # (2, 91)

  inputs = {
            'is_sdc': decoded_example['state/is_sdc'],
            'gt_future_states': gt_future_states, # (2, 91, 7)
            'gt_future_is_valid': gt_future_is_valid, # (2, 91)
            'past_states':past_states, # (2, 10, 7)
            'object_type': decoded_example['state/type'], # (2, )
            'x':x, # (2,)
            'y':y, # (2, )
            'yaw':yaw, # (2, )
            'scenario_id':scenario_id,
            'object_id': object_id}
  return inputs

def get_dataset(file_pattern, batch_size=16, shuffle=True):
  file_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4), cycle_length=8 if shuffle else 1)\
  .map(_parse, num_parallel_calls=8).batch(batch_size)
  return dataset

ot_feature_desc = {
    'state/type':
        tf.io.FixedLenFeature([2], tf.float32, default_value=None),
}

def _cyclist_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return tf.reduce_max(example['state/type']) == 3. 

def get_cyclist_dataset(file_pattern, batch_size=32, shuffle=True):
  file_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _cyclist_only(y)), cycle_length=8 if shuffle else 1)\
  .map(_parse, num_parallel_calls=8)
  return dataset.batch(batch_size)

def _ped_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return tf.reduce_max(example['state/type']) == 2. 

def get_ped_dataset(file_pattern, batch_size=32, shuffle=True):
  file_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _ped_only(y)), cycle_length=8 if shuffle else 1)\
  .map(_parse, num_parallel_calls=8)
  return dataset.batch(batch_size)

def _veh_only(data):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return tf.reduce_max(example['state/type']) == 1. 

def get_veh_dataset(file_pattern, batch_size=32, shuffle=True):
  file_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4).filter(lambda y: _veh_only(y)), cycle_length=8 if shuffle else 1)\
  .map(_parse, num_parallel_calls=8)
  return dataset.batch(batch_size)

def get_interaction_eval_dataset(data_type, eval_file_pattern, batch_size=32):
  if data_type == "cyclist":
    dataset = get_cyclist_dataset(eval_file_pattern, batch_size, shuffle=False)
  elif data_type == "ped":
    dataset = get_ped_dataset(eval_file_pattern, batch_size, shuffle=False)
  elif data_type == "veh":
    dataset = get_veh_dataset(eval_file_pattern, batch_size, shuffle=False)
  else:
    dataset = get_dataset(eval_file_pattern, batch_size, shuffle=False)
  return dataset

def get_dataset_for_clustering(file_pattern):
  file_dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = file_dataset\
  .interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(4), cycle_length=8)\
  .map(_parse_without_image, num_parallel_calls=8).batch(16)
  return dataset

def _object_type_only(data, object_type):
  example = tf.io.parse_single_example(data, ot_feature_desc)
  return tf.reduce_max(example['state/type']) == object_type 

def get_extended_dataset(train_file_pattern, validation_file_pattern, object_type):
  train_file_dataset = tf.data.Dataset.list_files(train_file_pattern)
  if validation_file_pattern:
    validation_file_dataset = tf.data.Dataset.list_files(validation_file_pattern)
    combined_dataset = train_file_dataset.concatenate(validation_file_dataset).shuffle(1100)
  else:
    combined_dataset = train_file_dataset
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

def get_weighted_dataset(train_file_pattern, validation_file_pattern, veh_val, ped_val, cyc_val, batch_size=16):
  veh_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=1)
  ped_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=2)
  cyc_dataset = get_extended_dataset(train_file_pattern, validation_file_pattern, object_type=3)
  return combine_datasets(veh_dataset, ped_dataset, cyc_dataset, veh_val, ped_val, cyc_val).batch(batch_size)