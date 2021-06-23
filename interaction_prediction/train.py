# This the interaction_prediction version!!!
import tensorflow as tf
from waymo_open_dataset.metrics.python import config_util_py as config_util
from .model import JointMOEModel
from .metrics import _default_metrics_config, MotionMetrics
from .loss import JointPredMultiPathLoss, JointPredMultiModalLoss
from .dataset import get_dataset, get_cyclist_dataset, get_weighted_dataset
from ..utils import DotDict, load_config_data, load_cnn_model
from ..train import load_model as load_marginal_model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml
import pprint

def load_model(weights_file = None, experiment_name='c8.yaml'):
  cfg = load_config_data(experiment_name)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  pprint.pprint(train_params)
  pprint.pprint(model_params)
  marginal_model = load_marginal_model(model_params.marginal_model_file, experiment_name = model_params.marginal_experiment_name)
  # the following are needed to prevent a model saving failure
  marginal_model.multi_modal_loss = None
  marginal_model.multi_modal_loss_tracker = None
  marginal_model.multi_path_loss = None
  marginal_model.loss_tracker = None

  if model_params.freeze_marginal:
    for layer in marginal_model.layers:
      layer.trainable = False
  
  K=model_params.num_modes if model_params.num_modes else 16
  dropout = model_params.dropout if model_params.dropout else 0.
  joint_conf_prediction = model_params.joint_conf_prediction if model_params.joint_conf_prediction is not None else True
  model = JointMOEModel(marginal_model, K=K, dropout=dropout, embedding_method=model_params.embedding_method, joint_conf_prediction=joint_conf_prediction)
  lr_schedule = tf.keras.experimental.CosineDecay(float(train_params.initial_lr), int(train_params.steps), alpha = float(train_params.alpha))

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  loss_fn = tf.keras.losses.MeanSquaredError()
  loss_tracker = tf.keras.metrics.Mean(name = 'loss')
  multi_modal_loss_tracker = tf.keras.metrics.Mean(name = 'mm_loss')
  metrics_config = _default_metrics_config()
  motion_metrics = MotionMetrics(metrics_config)
  metric_names = config_util.get_breakdown_names_from_motion_config(
      metrics_config)
  multi_modal_loss = JointPredMultiModalLoss()
  veh_centroids = np.load(model_params.veh_centroids).astype(np.float32)
  ped_centroids = np.load(model_params.ped_centroids).astype(np.float32)
  cyc_centroids = np.load(model_params.cyc_centroids).astype(np.float32)
  veh_weight = float(model_params.veh_weight) if model_params.veh_weight is not None else 1
  ped_weight = float(model_params.ped_weight) if model_params.ped_weight is not None else 1
  cyc_weight = float(model_params.cyc_weight) if model_params.cyc_weight is not None else 1
  cls_weight = float(model_params.cls_weight) if model_params.cls_weight is not None else 1
  marginal_loss_weight = float(model_params.marginal_loss_weight) if model_params.marginal_loss_weight is not None else 0

  multi_path_loss = JointPredMultiPathLoss(veh_centroids, ped_centroids, cyc_centroids, veh_weight=veh_weight, ped_weight=ped_weight, cyc_weight=cyc_weight, cls_weight=cls_weight, marginal_loss_weight=marginal_loss_weight)
  model.compile(optimizer, multi_path_loss, multi_modal_loss, loss_tracker, multi_modal_loss_tracker, motion_metrics, metrics_config)

  file_pattern = train_params.train_file_pattern
  dataset = get_dataset(file_pattern)

  eval_file_pattern = train_params.eval_file_pattern
  eval_dataset = get_dataset(eval_file_pattern)

  epochs = train_params.epochs
  save_freq = "epoch"
  if train_params.save_freq != "epoch":
    save_freq = int(train_params.save_freq)

  filepath = train_params.model_file_pattern
  callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              filepath,  monitor='val_loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq=save_freq),
      ]
  if weights_file:
    if weights_file[:5] != "drive":
      weights_file = "drive/MyDrive/Motion/MMM/" + weights_file
    model.fit(eval_dataset.take(1), epochs = 1)
    model.load_weights(weights_file)
  return model


def train(experiment_name='joint_c8.yaml', run_fit=True):
  cfg = load_config_data(experiment_name)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  model = load_model(train_params.initial_weights_file, experiment_name)
  batch_size = train_params.batch_size if train_params.batch_size else 16
  file_pattern = train_params.train_file_pattern
  if train_params.cyclist:
    dataset = get_cyclist_dataset(file_pattern, batch_size)
  elif train_params.weighted:
    dataset = get_weighted_dataset(file_pattern, train_params.validation_file_pattern, 0.33, 0.33, 0.33, batch_size)
  else:
    dataset = get_dataset(file_pattern, batch_size)

  eval_batch_size = train_params.eval_batch_size if train_params.eval_batch_size else 16
  eval_file_pattern = train_params.eval_file_pattern
  if train_params.cyclist:
    eval_dataset = get_cyclist_dataset(eval_file_pattern, eval_batch_size)
  else:
    eval_dataset = get_dataset(eval_file_pattern, eval_batch_size)
  print("eval_batch_size", eval_batch_size)

  epochs = train_params.epochs
  save_freq = "epoch"
  if train_params.save_freq != "epoch":
    save_freq = int(train_params.save_freq)

  filepath = train_params.model_file_pattern
  callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              filepath,  monitor='val_loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq=save_freq),
      ]
  if run_fit:
    if train_params.initial_weights_file:
      model.fit(eval_dataset.take(1), epochs = 1)
      model.load_weights(train_params.initial_weights_file)
    try:
      model.fit(dataset, epochs = epochs, validation_data = eval_dataset, callbacks = callbacks)
      model.save_weights(train_params.final_file)
    except KeyboardInterrupt:
      model.save_weights(train_params.final_file)
      return model
    return model
  else:
    model.fit(eval_dataset.take(1), epochs = 1)
    return model

def continue_train(model, experiment_name, initial_lr, epochs=20):
  cfg = load_config_data(experiment_name)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)

  file_pattern = train_params.train_file_pattern
  dataset = get_dataset(file_pattern)
  eval_file_pattern = train_params.eval_file_pattern
  eval_dataset = get_dataset(eval_file_pattern)
  save_freq = "epoch"
  if train_params.save_freq != "epoch":
    save_freq = int(train_params.save_freq)

  filepath = train_params.model_file_pattern
  callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              filepath,  
              monitor='val_loss', 
              save_best_only=False, 
              save_weights_only=True, 
              mode='auto', 
              save_freq=save_freq),
          LRRecorder(),
        ]
  optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
  model.compile(optimizer, model.multi_path_loss, model.multi_modal_loss, model.loss_tracker, model.multi_modal_loss_tracker, model.motion_metrics, model.metrics_config)
  try:
    model.fit(dataset, epochs = epochs, validation_data = eval_dataset, callbacks = callbacks)
    model.save_weights(train_params.final_file)
  except KeyboardInterrupt:
    return model