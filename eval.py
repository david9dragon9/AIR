import tensorflow as tf
from waymo_open_dataset.metrics.python import config_util_py as config_util
from .metrics import _default_metrics_config, MotionMetrics
from .loss import MultiModalLoss, MTMPLoss
from .dataset import get_dataset, get_deterministic_dataset, get_cyclist_dataset, get_ped_dataset, get_veh_dataset, get_eval_dataset
from MotionPrediction.interaction_prediction.dataset import get_interaction_eval_dataset
from .utils import DotDict, load_config_data, load_cnn_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import pprint
from .train import load_model


def evaluate(weights_file, experiment_file="c8.yaml", eval_file_pattern='drive/MyDrive/Motion/validation/images-00000-of-00150'):
  model = load_model(weights_file, experiment_file)
  cfg = load_config_data(experiment_file)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  batch_size = int(train_params.eval_batch_size) if train_params.eval_batch_size else 32
  if train_params.cyclist:
    dataset = get_cyclist_dataset(eval_file_pattern, batch_size)
  elif train_params.ped:
    dataset = get_ped_dataset(eval_file_pattern, batch_size)
  elif train_params.veh:
    dataset = get_veh_dataset(eval_file_pattern, batch_size)
  else:
    dataset = get_dataset(eval_file_pattern, batch_size)
  metric_names = config_util.get_breakdown_names_from_motion_config(
    model.metrics_config)
  model.motion_metrics.reset_states()
  for step, batch in enumerate(dataset):
    if step % 100 == 0:
      print("step", step)
    model.test_step(batch)
  metric_values = model.motion_metrics.result()
  for i, m in enumerate(
      ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
    for j, n in enumerate(metric_names):
        print('{}/{}: {}'.format(m, n, metric_values[i, j]))
  map_average = np.mean(metric_values[4, 0:9])
  print("map_average", map_average)
  print("cyc_map_average", np.mean(metric_values[4, 6:9]))

def display_mAP(metric_values):
  mAP_veh = np.mean(metric_values[4, 0:3])
  mAP_ped = np.mean(metric_values[4, 3:6])
  mAP_cyc = np.mean(metric_values[4, 6:])
  print("map/TYPE_VEHICLE", mAP_veh)
  print("map/TYPE_PEDESTRIAN", mAP_ped)
  print("map/TYPE_CYCLIST", mAP_cyc)
  return np.array([mAP_veh, mAP_ped, mAP_cyc])

# Metric-Loss Sensitivity Analysis
def dmAP_evaluate(weights_file, beta, gamma, experiment_file="moe_c16_2nd.yaml", eval_file_pattern='drive/MyDrive/Motion/validation/images-00000-of-00150'):
  model = load_model(weights_file, experiment_file)
  dataset = get_deterministic_dataset(eval_file_pattern)
  metric_names = config_util.get_breakdown_names_from_motion_config(
    model.metrics_config)
  model.motion_metrics.reset_states()
  model.loss_tracker.reset_states()
  model.conf_loss_tracker.reset_states()
  cls_motion_metrics = MotionMetrics(_default_metrics_config())
  reg_motion_metrics = MotionMetrics(_default_metrics_config())
  cls_loss_tracker = tf.keras.metrics.Mean(name = 'cls_loss')
  cls_conf_loss_tracker = tf.keras.metrics.Mean(name = 'cls_conf_loss')
  reg_loss_tracker = tf.keras.metrics.Mean(name = 'reg_loss')
  reg_conf_loss_tracker = tf.keras.metrics.Mean(name = 'reg_conf_loss')
  for step, batch in enumerate(dataset):
    model.dmAP_step(batch, cls_motion_metrics, reg_motion_metrics, cls_loss_tracker, cls_conf_loss_tracker, reg_loss_tracker, reg_conf_loss_tracker, beta = beta, gamma = gamma)
  metric_values = model.motion_metrics.result()
  cls_metric_values = cls_motion_metrics.result()
  reg_metric_values = reg_motion_metrics.result()

  print("default metric_values")
  base_mAPs = display_mAP(metric_values)

  print("cls metric values")
  cls_mAPs = display_mAP(cls_metric_values)

  print("reg metric values")
  reg_mAPs = display_mAP(reg_metric_values)

  base_conf_loss = model.conf_loss_tracker.result().numpy()
  base_reg_loss = model.loss_tracker.result().numpy() - base_conf_loss
  cls_conf_loss = cls_conf_loss_tracker.result().numpy()
  reg_reg_loss = reg_loss_tracker.result().numpy() - reg_conf_loss_tracker.result().numpy()

  print("multi_path_loss:", model.loss_tracker.result())
  print("conf_loss:", model.conf_loss_tracker.result())
  print("cls_loss:", cls_loss_tracker.result().numpy())
  print("cls_conf_loss", cls_conf_loss_tracker.result().numpy())
  print("reg_loss:", reg_loss_tracker.result().numpy())
  print("reg_conf_loss", reg_conf_loss_tracker.result().numpy())
  cls_delta = (cls_mAPs - base_mAPs)/(base_conf_loss - cls_conf_loss)
  reg_delta = (reg_mAPs - base_mAPs)/(base_reg_loss - reg_reg_loss)
  print(f"cls_delta:{cls_delta}\nreg_delta: {reg_delta}")

  return cls_delta, reg_delta