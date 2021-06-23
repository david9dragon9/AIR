import os
from .generate_data import process_one_raw_training_file, v2_process_one_raw_training_file

def process_validation(start=0, end=150):
  for i in range(start, end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord-0{i:04d}-of-00150 .'
    print(cmd)
    os.system(cmd)
    process_one_raw_training_file(f"validation_tfexample.tfrecord-0{i:04d}-of-00150", f"drive/MyDrive/Motion/validation/images-0{i:04d}-of-00150", include_images = True)
    rm_cmd = f'rm validation_tfexample.tfrecord-0{i:04d}-of-00150'
    print(rm_cmd)
    os.system(rm_cmd)

def v2_process_train(start=0, end=1000):
  for i in range(start, end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord-0{i:04d}-of-01000 .'
    print(cmd)
    os.system(cmd)
    v2_process_one_raw_training_file(f"training_tfexample.tfrecord-0{i:04d}-of-01000", f"drive/MyDrive/Motion/data/training_v2/images-0{i:04d}-of-01000")
    rm_cmd = f'rm training_tfexample.tfrecord-0{i:04d}-of-01000'
    print(rm_cmd)
    os.system(rm_cmd)

def v2_process_validation(start=0, end=150):
  for i in range(start, end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord-0{i:04d}-of-00150 .'
    print(cmd)
    os.system(cmd)
    v2_process_one_raw_training_file(f"validation_tfexample.tfrecord-0{i:04d}-of-00150", f"drive/MyDrive/Motion/data/validation_v2/images-0{i:04d}-of-00150")
    rm_cmd = f'rm validation_tfexample.tfrecord-0{i:04d}-of-00150'
    print(rm_cmd)
    os.system(rm_cmd)

def v2_process_testing(start=0, end=150):
  for i in range(start, end):
    cmd = f'gsutil cp gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/testing/testing_tfexample.tfrecord-0{i:04d}-of-00150 .'
    print(cmd)
    os.system(cmd)
    v2_process_one_raw_training_file(f"testing_tfexample.tfrecord-0{i:04d}-of-00150", f"drive/MyDrive/Motion/data/testing_v2/images-0{i:04d}-of-00150")
    rm_cmd = f'rm testing_tfexample.tfrecord-0{i:04d}-of-00150'
    print(rm_cmd)
    os.system(rm_cmd)
