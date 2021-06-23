# This is the interaction prediction version!!!
import tensorflow as tf
from waymo_open_dataset.metrics.python import config_util_py as config_util
from .model import JointMOEModel
from .metrics import _default_metrics_config, MotionMetrics
from .loss import JointPredMultiPathLoss, JointPredMultiModalLoss
from .dataset import get_dataset, get_cyclist_dataset, get_ped_dataset, get_veh_dataset, get_interaction_eval_dataset
from ..utils import DotDict, load_config_data, load_cnn_model
from MotionPrediction.interaction_prediction.ensemble import EnsembleModel
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import pprint
from .train import train, load_model


def evaluate(weights_file, experiment_file="joint_moe.yaml", eval_file_pattern='drive/MyDrive/Motion/interaction_data/validation_rr_v2/*', data_type=None):
  if weights_file[:5] != "drive":
    weights_file = "drive/MyDrive/Motion/MMM/" + weights_file
  model = train(experiment_file, run_fit=False)
  cfg = load_config_data(experiment_file)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  model.load_weights(weights_file)
  batch_size = int(train_params.eval_batch_size) if train_params.eval_batch_size else 16
  dataset = get_interaction_eval_dataset(data_type, eval_file_pattern, batch_size)
  metric_names = config_util.get_breakdown_names_from_motion_config(
    model.metrics_config)
  model.motion_metrics.reset_states()
  for step, batch in enumerate(dataset):
    if step % 100 == 0:
      print(step)
    model.test_step(batch)
  metric_values = model.motion_metrics.result()
  for i, m in enumerate(
      ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
    for j, n in enumerate(metric_names):
        print('{}/{}: {}'.format(m, n, metric_values[i, j]))
  map_average = np.mean(metric_values[4, 0:9])
  print("map_average", map_average)
  print("veh_map_average", np.mean(metric_values[4, 0:3]))
  print("ped_map_average", np.mean(metric_values[4, 3:6]))
  print("cyc_map_average", np.mean(metric_values[4, 6:9]))

def evaluate_ensemble(weights_files, experiment_files, num_modes, data_type = None, eval_file_pattern='drive/MyDrive/Motion/validation/images-00000-of-00150'):
  models = []
  for i in range(len(weights_files)):
    print(f"-----loading model {i}")
    if weights_files[i][:5] != "drive": 
      weights_files[i] = "drive/MyDrive/Motion/MMM/" + weights_files[i]
    models.append(load_model(weights_files[i], experiment_files[i]))
  model = EnsembleModel(models, num_modes)
  cfg = load_config_data(experiment_files[0])
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  batch_size = int(train_params.eval_batch_size) if train_params.eval_batch_size else 32
  dataset = get_interaction_eval_dataset(data_type, eval_file_pattern, batch_size)
  metric_names = config_util.get_breakdown_names_from_motion_config(
    model.metrics_config)
  model.motion_metrics.reset_states()
  for step, batch in enumerate(dataset):
    if step % 100 == 0:
      print("step", step)
    model.prev_test_step(batch)
  metric_values = model.motion_metrics.result()
  for i, m in enumerate(
      ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
    for j, n in enumerate(metric_names):
        print('{}/{}: {}'.format(m, n, metric_values[i, j]))
  map_average = np.mean(metric_values[4, 0:9])
  print("map_average", map_average)
  print("cyc_map_average", np.mean(metric_values[4, 6:9]))
