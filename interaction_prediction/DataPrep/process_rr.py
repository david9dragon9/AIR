# This is the rerasterization version!!!
import os
from MotionPrediction.interaction_prediction.DataPrep.generate_data import process_one_raw_training_file_for_rr
from MotionPrediction.train import load_model

def process_validation(start=0, end=150, weights_file= "drive/MyDrive/Motion/MMM/cls_weight_lr3_2.01-7.6130.hdf5", yaml_file="cls_weight_lg_lr_2.yaml"):
  model = load_model(weights_file, yaml_file)
  for i in range(start,end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation_interactive/validation_interactive_tfexample.tfrecord-0{i:04d}-of-00150 .'
    print(cmd)
    os.system(cmd)
    process_one_raw_training_file_for_rr(
      f"validation_interactive_tfexample.tfrecord-0{i:04d}-of-00150",
      f"drive/MyDrive/Motion/interaction_data/validation_yellow/images-0{i:04d}-of-00150",
      f"drive/MyDrive/Motion/interaction_data/validation_rr/images-0{i:04d}-of-00150",
      model,
      )
    rm_cmd = f'rm validation_interactive_tfexample.tfrecord-0{i:04d}-of-00150'
    print(rm_cmd)
    os.system(rm_cmd)

def process_train(start=0, end=1000, weights_file= "drive/MyDrive/Motion/MMM/cls_weight_lr3_2.01-7.6130.hdf5", yaml_file="cls_weight_lg_lr_2.yaml"):
  model = load_model(weights_file, yaml_file)
  for i in range(start,end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord-0{i:04d}-of-01000 .'
    print(cmd)
    os.system(cmd)
    process_one_raw_training_file_for_rr(
      f"training_tfexample.tfrecord-0{i:04d}-of-01000", 
      f"drive/MyDrive/Motion/interaction_data/training_yellow/images-0{i:04d}-of-01000",
      f"drive/MyDrive/Motion/interaction_data/training_rr/images-0{i:04d}-of-01000",
      model
      )
    rm_cmd = f'rm training_tfexample.tfrecord-0{i:04d}-of-01000'
    print(rm_cmd)
    os.system(rm_cmd)

def process_testing(start=0, end=150, weights_file= "drive/MyDrive/Motion/MMM/cls_weight_lr3_2.01-7.6130.hdf5", yaml_file="cls_weight_lg_lr_2.yaml"):
  model = load_model(weights_file, yaml_file)
  for i in range(start,end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/testing_interactive/testing_interactive_tfexample.tfrecord-0{i:04d}-of-00150 .'
    print(cmd)
    os.system(cmd)
    process_one_raw_training_file_for_rr(
      f"testing_interactive_tfexample.tfrecord-0{i:04d}-of-00150", 
      f"drive/MyDrive/Motion/interaction_data/testing_yellow/images-0{i:04d}-of-00150",
      f"drive/MyDrive/Motion/interaction_data/testing_rr/images-0{i:04d}-of-00150",
      model
      )
    rm_cmd = f'rm testing_interactive_tfexample.tfrecord-0{i:04d}-of-00150'
    print(rm_cmd)
    os.system(rm_cmd)