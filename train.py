import tensorflow as tf
from waymo_open_dataset.metrics.python import config_util_py as config_util
from .model import MOEModelWithVariableClusters
from .metrics import _default_metrics_config, MotionMetrics
from .loss import MultiModalLoss, MTMPLoss
from .dataset import get_dataset, get_cyclist_dataset, get_ped_dataset, get_veh_dataset, get_weighted_dataset
from .utils import DotDict, load_config_data, load_cnn_model, LRRecorder
import tensorflow as tf
import numpy as np
import pprint

def load_model(weights_file = None, experiment_name='c8.yaml'):
  backbone = load_cnn_model()
  cfg = load_config_data(experiment_name)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  if model_params.backbone_weights and not weights_file:
    print(f"loading backbone weights from {model_params.backbone_weights}...")
    backbone.load_weights(model_params.backbone_weights)
    print(f"done loading!")
  if model_params.freeze_backbone == "partial":
    for layer in backbone.layers:
      if layer.name == 'block2a_dwconv_pad':
        break
      layer.trainable = False
  elif model_params.freeze_backbone:
    backbone.trainable = False
  num_ps_units = train_params.num_ps_units if "num_ps_units" in train_params else None
  num_future_steps = 80  
  veh_centroids = np.load(model_params.veh_centroids)
  ped_centroids = np.load(model_params.ped_centroids)
  cyc_centroids = np.load(model_params.cyc_centroids)
  veh_modes = len(veh_centroids)
  ped_modes = len(ped_centroids)
  cyc_modes = len(cyc_centroids)
  num_knots = model_params.num_knots if model_params.num_knots is not None else 8
  dropout = model_params.dropout if model_params.dropout is not None else 0.
  model = MOEModelWithVariableClusters(80, backbone, num_modes = model_params.num_modes, num_knots = num_knots, veh_modes = veh_modes, ped_modes = ped_modes, cyc_modes = cyc_modes, loss_type = 'multi_path_loss', num_ps_units = num_ps_units, dropout=dropout)
  lr_schedule = tf.keras.experimental.CosineDecay(float(train_params.initial_lr), int(train_params.steps), alpha = float(train_params.alpha))

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  loss_fn = tf.keras.losses.MeanSquaredError()
  loss_tracker = tf.keras.metrics.Mean(name = 'loss')
  multi_modal_loss_tracker = tf.keras.metrics.Mean(name = 'mm_loss')
  metrics_config = _default_metrics_config()
  motion_metrics = MotionMetrics(metrics_config)
  metric_names = config_util.get_breakdown_names_from_motion_config(
      metrics_config)
  multi_modal_loss = MultiModalLoss(num_future_steps=num_future_steps)
  veh_centroids = np.load(model_params.veh_centroids)
  ped_centroids = np.load(model_params.ped_centroids)
  cyc_centroids = np.load(model_params.cyc_centroids)
  cls_weight = float(model_params.cls_weight) if model_params.cls_weight is not None else 1.0
  reg_weight = float(model_params.reg_weight) if model_params.reg_weight is not None else 1.0
  ped_weight = float(model_params.ped_weight) if model_params.ped_weight is not None else 1.0
  cyc_weight = float(model_params.cyc_weight) if model_params.cyc_weight is not None else 1.0
  multi_path_loss = MTMPLoss(veh_centroids, ped_centroids, cyc_centroids, cls_weight, reg_weight, ped_weight, cyc_weight, num_future_steps)
  model.compile(optimizer, multi_path_loss, multi_modal_loss, loss_tracker, multi_modal_loss_tracker, motion_metrics, metrics_config)

  if weights_file:
    eval_file_pattern = train_params.eval_file_pattern
    eval_dataset = get_dataset(eval_file_pattern, batch_size=1)
    model.fit(eval_dataset.take(1), epochs = 1)
    model.load_weights(weights_file)
  return model

def train(experiment_name='c8.yaml'):
  cfg = load_config_data(experiment_name)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  pprint.pprint(train_params)
  pprint.pprint(model_params)

  model = load_model(train_params.initial_weights_file, experiment_name=experiment_name)

  file_pattern = train_params.train_file_pattern
  batch_size = int(train_params.batch_size) if train_params.batch_size is not None else 32
  if train_params.cyclist:
    dataset = get_cyclist_dataset(file_pattern, batch_size=batch_size)
  elif train_params.ped:
    dataset = get_ped_dataset(file_pattern, batch_size=batch_size)
  elif train_params.veh:
    dataset = get_veh_dataset(file_pattern, batch_size=batch_size)
  elif train_params.weighted:
    dataset = get_weighted_dataset(file_pattern, train_params.validation_file_pattern, 0.33, 0.33, 0.33, batch_size)
  else:
    dataset = get_dataset(file_pattern, batch_size=batch_size, filter_valid=True)

  eval_file_pattern = train_params.eval_file_pattern
  if train_params.cyclist:
    eval_dataset = get_cyclist_dataset(eval_file_pattern, batch_size=batch_size)
  elif train_params.ped:
    eval_dataset = get_ped_dataset(eval_file_pattern, batch_size=batch_size)
  elif train_params.veh:
    eval_dataset = get_veh_dataset(eval_file_pattern, batch_size=batch_size)
  else:
    eval_dataset = get_dataset(eval_file_pattern, batch_size=batch_size)

  epochs = train_params.epochs
  save_freq = "epoch"
  if train_params.save_freq != "epoch":
    save_freq = int(train_params.save_freq)

  filepath = train_params.model_file_pattern
  callbacks = [
          tf.keras.callbacks.ModelCheckpoint(
              filepath,  monitor='val_loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq=save_freq),
          LRRecorder()
      ]
  try:
    model.fit(dataset, epochs = epochs, validation_data = eval_dataset, callbacks = callbacks)
  except KeyboardInterrupt:
    return model
  model.save_weights(train_params.final_file)

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
  model.fit(dataset, epochs = epochs, validation_data = eval_dataset, callbacks = callbacks)
  model.save_weights(train_params.final_file)
