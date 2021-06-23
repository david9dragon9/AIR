import math
import os
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from tensorflow_graphics.math.interpolation import bspline

from .utils import transform_matrix, truncate_predictions
from .utils import transform_points, draw_trajectory, pad_to_shape

class Head(tf.keras.layers.Layer):
  def __init__(self, centroids, num_knots=8, num_future_steps=80):
    super(Head, self).__init__()
    self.centroids = tf.cast(tf.convert_to_tensor(centroids), tf.float32)
    self.num_modes = len(centroids)
    self.num_knots = num_knots
    self.num_future_steps = num_future_steps
    self.softmax = tf.keras.layers.Softmax()

  def call(self, inputs, x, y, yaw, past_ego_x=None, past_ego_y=None):
    """
    Parameters:
      inputs: (B, num_modes + num_modes*num_knots*2)
      x: (B, 1)
      past_ego_x: (B, 10)
    Returns:
      trajectories: (B, num_modes, num_future_steps, 2)
      confidences: (B, num_modes)
    """

    c = tf.math.cos(yaw)
    s = tf.math.sin(yaw)

    pred = inputs[:, self.num_modes:] # (B, 480)
    knots = tf.reshape(pred, (-1, self.num_modes, 2, self.num_knots)) # (B, 3, 2, 8)
    max_pos = self.num_knots - 3 # + 1
    positions = tf.expand_dims(tf.range(start = 0.0, limit = max_pos, delta = max_pos/self.num_future_steps, dtype= knots.dtype), axis = -1)
    spline = bspline.interpolate(knots, positions, 3, False)
    spline = tf.squeeze(spline, axis = 1)
    pred = tf.transpose(spline, perm = [1,2,0,3]) # (B, K, 80, 2)

    centroid_bias = tf.reshape(self.centroids, [-1, self.num_modes, 80, 2])

    pred = pred + centroid_bias

    pred = tf.reshape(pred, [-1, self.num_modes*self.num_future_steps, 2])

    pred_x = c*pred[...,0] - s*pred[...,1] + x
    pred_y = s*pred[...,0] + c*pred[...,1] + y
    pred = tf.stack([pred_x, pred_y], -1)
    pred = tf.reshape(pred, [-1, self.num_modes, self.num_future_steps, 2], name="PRED")

    confidences = inputs[:, :self.num_modes]
    confidences = self.softmax(confidences)
    return pred, confidences, knots

class MOEModelWithVariableClusters(tf.keras.Model):
  def __init__(self, num_future_steps, backbone, num_modes=3, loss_type = 'multi_path_loss', num_knots = 8, num_ps_units = None, veh_modes = 16, ped_modes = 8, cyc_modes = 8, dropout=0.):
    super().__init__()
    self._num_knots = num_knots
    self._loss_type = loss_type
    self._num_future_steps = num_future_steps
    self.backbone = backbone
    self.pooling = tf.keras.layers.GlobalAveragePooling2D()
    if dropout > 0:
      self.shared_ffn = tf.keras.Sequential(
              [tf.keras.layers.Dropout(dropout),
              tf.keras.layers.Dense(300, activation="relu")])
    else:
      self.shared_ffn = tf.keras.Sequential(
              [tf.keras.layers.Dense(300, activation="relu")])
    self.veh_modes = veh_modes
    self.ped_modes = ped_modes
    self.cyc_modes = cyc_modes
    self.num_modes = max(veh_modes, ped_modes, cyc_modes)
    if dropout > 0:
      self.veh_fc = tf.keras.Sequential(
              [tf.keras.layers.Dropout(dropout),
              tf.keras.layers.Dense(veh_modes * 2 * num_knots + veh_modes)])
      self.ped_fc = tf.keras.Sequential(
              [tf.keras.layers.Dropout(dropout),
              tf.keras.layers.Dense(ped_modes * 2 * num_knots + ped_modes)])
      self.cyc_fc = tf.keras.Sequential(
              [tf.keras.layers.Dropout(dropout),
              tf.keras.layers.Dense(cyc_modes * 2 * num_knots + cyc_modes)])
    else:
      self.veh_fc = tf.keras.layers.Dense(veh_modes * 2 * num_knots + veh_modes)
      self.ped_fc = tf.keras.layers.Dense(ped_modes * 2 * num_knots + ped_modes)
      self.cyc_fc = tf.keras.layers.Dense(cyc_modes * 2 * num_knots + cyc_modes)
    self.softmax = tf.keras.layers.Softmax()
    if num_ps_units is None:
      num_ps_units = num_future_steps*2*num_modes + num_modes

    self.veh_ego_ffn = tf.keras.Sequential([tf.keras.layers.Dense(num_ps_units, activation = 'relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(veh_modes * 2 * num_knots + veh_modes)])
    self.ped_ego_ffn = tf.keras.Sequential([tf.keras.layers.Dense(num_ps_units, activation = 'relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(ped_modes * 2 * num_knots + ped_modes)])
    self.cyc_ego_ffn = tf.keras.Sequential([tf.keras.layers.Dense(num_ps_units, activation = 'relu'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(cyc_modes * 2 * num_knots + cyc_modes)])

  def call(self, image, x=None, y=None, yaw=None, past_states = None, object_type = None, is_sdc = None, training=True):
    image = tf.cast(image, tf.float32)
    c = tf.math.cos(yaw) # (B, 1)
    s = tf.math.sin(yaw) # (B, 1)
    embeddings = self.backbone(image, training=training)
    embeddings = embeddings[:,1:-1,1:-1,:]
    embeddings = self.pooling(embeddings) # Outputs (B,2048)
    past_states = tf.squeeze(past_states, axis = 1)
    past_x = past_states[:, :, 0] # (B, 10)
    past_y = past_states[:, :, 1] # (B, 10)
    past_x_hat = past_x - x # (B, 10)
    past_y_hat = past_y - y # (B, 10)
    past_ego_x = c * past_x_hat + s * past_y_hat # (B, 10)
    past_ego_y = -s * past_x_hat + c * past_y_hat # (B, 10)

    past_length = tf.reduce_mean(past_states[:, :, 2], axis = 1, keepdims = True)
    past_width = tf.reduce_mean(past_states[:, :, 3], axis = 1, keepdims = True)
    past_yaw = past_states[:, :, 4] - yaw
    past_vel_x = past_states[:, :, 5]
    past_vel_y = past_states[:, :, 6]
    past_vel_x = c * past_vel_x + s * past_vel_y # (B, 10)
    past_vel_y = -s * past_vel_x + c * past_vel_y # (B, 10)
    # is_sdc has shape (B, 1)

    past_states = tf.concat([past_ego_x, past_ego_y, past_length, past_width, past_yaw, past_vel_x, past_vel_y, tf.cast(is_sdc, tf.float32)], 1)

    is_veh = tf.cast(object_type == 1, tf.float32) # (B, 1)
    is_ped = tf.cast(object_type == 2, tf.float32)
    is_cyc = 1 - is_veh - is_ped

    shared_output = self.shared_ffn(embeddings, training=training)
    veh_output = self.veh_fc(shared_output, training=training) + self.veh_ego_ffn(past_states, training = training)
    ped_output = self.ped_fc(shared_output, training=training) + self.ped_ego_ffn(past_states, training = training)
    cyc_output = self.cyc_fc(shared_output, training=training) + self.cyc_ego_ffn(past_states, training = training)

    veh_pred, veh_conf, veh_knots = self.veh_head(veh_output, x, y, yaw)
    ped_pred, ped_conf, ped_knots = self.ped_head(ped_output, x, y, yaw)
    cyc_pred, cyc_conf, cyc_knots = self.cyc_head(cyc_output, x, y, yaw)

    veh_pred = pad_to_shape(veh_pred, [-1, self.num_modes, 80, 2])
    ped_pred = pad_to_shape(ped_pred, [-1, self.num_modes, 80, 2])
    cyc_pred = pad_to_shape(cyc_pred, [-1, self.num_modes, 80, 2])

    veh_conf = pad_to_shape(veh_conf, [-1, self.num_modes])
    ped_conf = pad_to_shape(ped_conf, [-1, self.num_modes])
    cyc_conf = pad_to_shape(cyc_conf, [-1, self.num_modes])

    pred = is_veh[:, :, tf.newaxis, tf.newaxis]*veh_pred +\
           is_ped[:, :, tf.newaxis, tf.newaxis]*ped_pred +\
           is_cyc[:, :, tf.newaxis, tf.newaxis]*cyc_pred

    conf = is_veh*veh_conf + is_ped*ped_conf + is_cyc*cyc_conf

    return (pred, conf, shared_output, embeddings)

  def compile(self, optimizer, multi_path_loss, multi_modal_loss, loss_tracker, multi_modal_loss_tracker, motion_metrics, metrics_config):
    super().compile(optimizer = optimizer, loss=multi_path_loss, metrics = [motion_metrics])
    self.multi_path_loss = multi_path_loss
    self.multi_modal_loss = multi_modal_loss
    self.multi_modal_loss_tracker = multi_modal_loss_tracker
    self.motion_metrics = motion_metrics
    self.metrics_config = metrics_config
    self.loss_tracker = loss_tracker
    self.conf_loss_tracker = tf.keras.metrics.Mean(name = 'conf_loss')
    self.veh_centroids = tf.convert_to_tensor(self.multi_path_loss.veh_centroids)
    self.ped_centroids = tf.convert_to_tensor(self.multi_path_loss.ped_centroids)
    self.cyc_centroids = tf.convert_to_tensor(self.multi_path_loss.cyc_centroids)
    self.veh_head = Head(self.veh_centroids, num_knots = self._num_knots)
    self.ped_head = Head(self.ped_centroids, num_knots = self._num_knots)
    self.cyc_head = Head(self.cyc_centroids, num_knots = self._num_knots)


  def train_step(self, inputs):
    with tf.GradientTape() as tape:
      gt_trajectory = tf.squeeze(inputs['gt_future_states'], axis = 1)
      gt_is_valid = inputs['gt_future_is_valid']
      gt_is_valid = tf.squeeze(gt_is_valid, axis = [1]) # (B, 91)
      prediction_start = self.metrics_config.track_history_samples + 1
      gt_targets = gt_trajectory[:, prediction_start:, :2] # (B, 80, 2)
      weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32) # (B, 80,)
      
      images = inputs['image']
      x = tf.squeeze(inputs['x'], axis = [1])
      y = tf.squeeze(inputs['y'], axis = [1])
      yaw = tf.squeeze(inputs['yaw'], axis = [1])
      past_states = inputs['past_states']
      object_type = inputs['object_type']
      pred_trajectory, confidences, _, _ = self(images, x=x, y=y, yaw=yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training=True)
      if self._loss_type == 'multi_modal_loss':
        loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences, object_type = object_type)
      else:
        loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, confidences, x, y, yaw, object_type = object_type)
    self.loss_tracker.update_state(loss)
    self.conf_loss_tracker.update_state(conf_loss)
    mm_loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences, object_type = object_type)
    self.multi_modal_loss_tracker.update_state(mm_loss)
    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    
    object_type = tf.squeeze(inputs['object_type'])
    pred_score = confidences
    return {"loss": self.loss_tracker.result(), "mm_loss" : self.multi_modal_loss_tracker.result(), "conf_loss": self.conf_loss_tracker.result()}

  def test_step(self, inputs):
    images = inputs['image']
    gt_trajectory = tf.squeeze(inputs['gt_future_states'], axis = 1)
    gt_is_valid = inputs['gt_future_is_valid']
    gt_is_valid = tf.squeeze(gt_is_valid, axis = [1])
    prediction_start = self.metrics_config.track_history_samples + 1
    gt_targets = gt_trajectory[:, prediction_start:, :2]
    weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)    
    x = tf.squeeze(inputs['x'], axis = [1])
    y = tf.squeeze(inputs['y'], axis = [1])
    yaw = tf.squeeze(inputs['yaw'], axis = [1])
    past_states = inputs['past_states']
    object_type = inputs['object_type']
    
    pred_trajectory, confidences, _, _ = self(images, x=x, y=y, yaw=yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training=False)
    if self._loss_type == 'multi_modal_loss':
      loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences, object_type = object_type)
    else:
      loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, confidences, x, y, yaw, object_type = object_type)
    self.loss_tracker.update_state(loss)
    self.conf_loss_tracker.update_state(conf_loss)
    mm_loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences, object_type = object_type)
    self.multi_modal_loss_tracker.update_state(mm_loss)

    object_type = tf.squeeze(inputs['object_type'])
    pred_score = confidences
    pred_trajectory, pred_score, _ = truncate_predictions(pred_trajectory, pred_score)
    self.motion_metrics.update_state(
        pred_trajectory,
        pred_score,
        gt_trajectory,
        gt_is_valid,
        object_type)
    return {"loss": self.loss_tracker.result(), "mm_loss": self.multi_modal_loss_tracker.result(), "conf_loss": self.conf_loss_tracker.result()}

  def predict_step(self, inputs, truncate=True):
    """
      Parameters:
        inputs: A dictionary mapping from parsed feature names to batched tensors
      Returns: 
        trajectories: tensor of shape(B, 3, 16, 2)
        confidences: tensor of shape (B, 3)
    """
    past_states = inputs['past_states'] # (B, 10, 2)
    images = inputs['image'] # (B, 224, 448, 3)
    x = tf.squeeze(inputs['x'], axis = [1])
    y = tf.squeeze(inputs['y'], axis = [1])
    yaw = tf.squeeze(inputs['yaw'], axis = [1])
    object_type = inputs['object_type']

    pred_trajectory, confidences, _, _ = self(images, x, y, yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training = False) # (B, 3, 80, 2)
    if truncate:
      interval = 5
      prediction_trajectory = pred_trajectory[:, :, (interval - 1)::interval]
      prediction_trajectory, confidences, _ = truncate_predictions(prediction_trajectory, confidences)
    else:
      prediction_trajectory = pred_trajectory
    return prediction_trajectory, confidences

  def inspect_step(self, inputs, draw_pred=False):
    past_states = inputs['past_states']
    images = inputs['image']
    gt_trajectory = tf.squeeze(inputs['gt_future_states'], axis = 1)
    gt_is_valid = inputs['gt_future_is_valid'] # (B, 80)
    gt_is_valid = tf.squeeze(gt_is_valid, axis = [1])
    prediction_start = self.metrics_config.track_history_samples + 1
    gt_targets = gt_trajectory[:, prediction_start:, :2] # (B, 80, 2)
    weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32) 
    avails = weights.numpy().astype(np.bool)   
    x = tf.squeeze(inputs['x'], axis = [1])
    y = tf.squeeze(inputs['y'], axis = [1])
    yaw = tf.squeeze(inputs['yaw'], axis = [1])
    image = tf.squeeze(inputs['image']).numpy()
  
    if draw_pred:
      object_type = inputs['object_type']
      orig_pred_trajectory, orig_confidences, _, knots = self(images, x, y, yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training = False) # (B, 3, 80, 2)
      loss, conf_loss, assignments_mask = self.multi_path_loss.call(gt_targets, weights, orig_pred_trajectory, orig_confidences, x, y, yaw, object_type = object_type, reduce_mean=False)
      pred_trajectory, confidences, indices = truncate_predictions(orig_pred_trajectory, orig_confidences, k=1)
      loss_indices = tf.where(loss > 0).numpy()
      if len(loss_indices) > 0:
        loss_indices = loss_indices[0]
      image = image.astype(np.float32)*1
      pred_trajectory = pred_trajectory.numpy()
      for example_index in loss_indices:
        assignment = tf.where(assignments_mask[example_index] == 1).numpy()[0, 0] # (K,)
        print(example_index, assignment)
        print("object_type", object_type[example_index].numpy())
        print("loss:", loss[example_index].numpy())
        print("conf_loss:", conf_loss[example_index].numpy())
        print("confidence:", confidences[example_index].numpy())
        print("assignment_confidence:", orig_confidences[example_index, assignment].numpy())
        print("image[example_index].shape", image[example_index].shape)
        current_x = tf.squeeze(x[example_index]).numpy()
        current_y = tf.squeeze(y[example_index]).numpy()
        current_yaw = tf.squeeze(yaw[example_index]).numpy()
        world_to_image = transform_matrix(current_x, current_y, current_yaw)
        trajectories = np.concatenate([pred_trajectory[example_index], orig_pred_trajectory[example_index, assignment].numpy()[np.newaxis, :, :]], 0)
        draw_trajectory(image[example_index], world_to_image, trajectory = trajectories, gt_trajectory = tf.squeeze(gt_targets[example_index]).numpy(), avails=avails[example_index])

  def dmAP_step(self, inputs, cls_motion_metrics, reg_motion_metrics, cls_loss_tracker, cls_conf_loss_tracker, reg_loss_tracker, reg_conf_loss_tracker, beta=0.5, gamma = 1.):
    images = inputs['image']
    gt_trajectory = tf.squeeze(inputs['gt_future_states'], axis = 1)
    gt_is_valid = inputs['gt_future_is_valid']
    gt_is_valid = tf.squeeze(gt_is_valid, axis = [1])
    prediction_start = self.metrics_config.track_history_samples + 1
    gt_targets = gt_trajectory[:, prediction_start:, :2]
    weights = tf.cast(gt_is_valid[:, prediction_start:], tf.float32)    
    x = tf.squeeze(inputs['x'], axis = [1])
    y = tf.squeeze(inputs['y'], axis = [1])
    yaw = tf.squeeze(inputs['yaw'], axis = [1])
    past_states = inputs['past_states']
    object_type = inputs['object_type']
    squeezed_object_type = tf.squeeze(inputs['object_type'])
    
    pred_trajectory, confidences, _ = self(images, x=x, y=y, yaw=yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training=False)
    loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, confidences, x, y, yaw, object_type = object_type)
    self.loss_tracker.update_state(loss)
    self.conf_loss_tracker.update_state(conf_loss)

    pred_score = confidences
    truncated_pred_trajectory, truncated_pred_score = truncate_predictions(pred_trajectory, pred_score)
    self.motion_metrics.update_state(
        truncated_pred_trajectory,
        truncated_pred_score,
        gt_trajectory,
        gt_is_valid,
        squeezed_object_type)
    gt = gt_targets
    avails = weights
    veh_centroids = self.multi_path_loss.veh_centroids
    ped_centroids = self.multi_path_loss.ped_centroids
    cyc_centroids = self.multi_path_loss.cyc_centroids
    c = tf.math.cos(yaw) # (B, 1)
    s = tf.math.sin(yaw) # (B, 1)
    x_hat = gt[:, :, 0] - x # (B, 80)
    y_hat = gt[:, :, 1] - y # (B, 80)
    gt_ego_x = c * x_hat + s * y_hat # (B, 80)
    gt_ego_y = -s * x_hat + c * y_hat # (B, 80)

    gt = tf.reshape(gt, [-1, 1, 80, 2])
    avails = tf.reshape(avails, [-1, 1, 80, 1])

    gt_ego = tf.stack([gt_ego_x, gt_ego_y], axis = -1) # (B, 80, 2)
    gt_ego = tf.reshape(gt_ego, [-1, 1, 80, 2])
      
    is_veh = object_type == 1 # object_type[:, 1] # (B,) object_type == 1
    is_ped = object_type == 2 # (B,)
    is_cyc = tf.math.logical_not(tf.math.logical_or(is_veh, is_ped))
    object_type_mask = tf.concat([is_veh, is_ped, is_cyc], axis=-1) # (B, 3)
  
    veh_centroids = tf.reshape(veh_centroids, [1, -1, 80, 2])
    distance = ((gt_ego - veh_centroids)**2)*avails # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    veh_assignments = tf.argmin(distance, axis = 1) # (B,)
    veh_assignments_mask = tf.one_hot(veh_assignments, self.num_modes) # (B, K)

    ped_centroids = tf.reshape(ped_centroids, [1, -1, 80, 2])
    distance = ((gt_ego - ped_centroids)**2)*avails # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    ped_assignments = tf.argmin(distance, axis = 1) # (B,)
    ped_assignments_mask = tf.one_hot(ped_assignments, self.num_modes) # (B, K)

    cyc_centroids = tf.reshape(cyc_centroids, [1, -1, 80, 2])
    distance = ((gt_ego - cyc_centroids)**2)*avails # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    cyc_assignments = tf.argmin(distance, axis = 1) # (B,)
    cyc_assignments_mask = tf.one_hot(cyc_assignments, self.num_modes) # (B, K)
    
    assignments_mask = tf.stack([veh_assignments_mask, ped_assignments_mask, cyc_assignments_mask], axis=1)
    assignments_mask = tf.boolean_mask(assignments_mask, object_type_mask) # (B, K)
    
    # Setting confidences to be the correct clustering assignments
    pred_score = assignments_mask * beta + confidences * (1 - beta)
    loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, pred_score, x, y, yaw, object_type = object_type)
    cls_loss_tracker.update_state(loss)
    cls_conf_loss_tracker.update_state(conf_loss)

    truncated_pred_trajectory, truncated_pred_score = truncate_predictions(pred_trajectory, pred_score)
    cls_motion_metrics.update_state(
        truncated_pred_trajectory,
        truncated_pred_score,
        gt_trajectory,
        gt_is_valid,
        squeezed_object_type)
    
    pred_trajectory = assignments_mask[:, :, tf.newaxis, tf.newaxis] * (gt * gamma + (1-gamma)*pred_trajectory) + (1-assignments_mask[:, :, tf.newaxis, tf.newaxis])*pred_trajectory
    pred_score = confidences
    loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, pred_score, x, y, yaw, object_type = object_type)
    reg_loss_tracker.update_state(loss)
    reg_conf_loss_tracker.update_state(conf_loss)

    truncated_pred_trajectory, truncated_pred_score = truncate_predictions(pred_trajectory, pred_score)
    reg_motion_metrics.update_state(
        truncated_pred_trajectory,
        truncated_pred_score,
        gt_trajectory,
        gt_is_valid,
        squeezed_object_type)

  @property
  def metrics(self):
    return [self.loss_tracker, self.multi_modal_loss_tracker, self.conf_loss_tracker]