# This is the interaction prediction version of the model!
import math
import os
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import itertools
import tensorflow as tf

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2
from tensorflow_graphics.math.interpolation import bspline

from ..utils import transform_matrix, truncate_predictions, pad_to_shape
from ..utils import transform_points, draw_trajectory, cartesian_product, confidence_cartesian_product
from ..utils import EgoToWorld, WorldToEgo

class JointMOEModel(tf.keras.Model):
  def __init__(self, marginal_model, K=16, dropout=0., embedding_method=None, joint_conf_prediction=True):
    super().__init__()
    self.marginal_model = marginal_model
    if joint_conf_prediction:
      self.softmax = tf.keras.layers.Softmax()
      self.shared_conf_ffn = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(300, activation = 'relu')])
      self.vv_conf_fc = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(K*K)])
      self.vs_conf_fc = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(K*K)])
      self.vp_conf_fc = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(K*K)])
      self.vc_conf_fc = tf.keras.Sequential([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(K*K)])
    self.K = K
    self.embedding_method = embedding_method
    self.joint_conf_prediction = joint_conf_prediction
    
  def call(self, image = None, x = None, y = None, yaw = None, past_states = None, object_type = None, is_sdc = None, training = True):
    """
      Call Parameters:
        image: (B, 2, 224, 448, 3)
        x: (B, 2, 1)
        y: (B, 2, 1)
        yaw: (B, 2, 1)
        past_states: (B, 2, 10, 7)
        object_type: (B, 2)
        is_sdc: (B, 2)
      Returns:
        trajectories: (B, 64, 2, 80, 2)
        confidences: (B, 64)
    """
    image = tf.reshape(image, [-1, 224,448,3])
    x = tf.reshape(x, [-1, 1])
    y = tf.reshape(y, [-1, 1])
    yaw = tf.reshape(yaw, [-1, 1])
    is_sdc = tf.reshape(is_sdc, [-1, 1])
    past_states = tf.reshape(past_states, [-1, 1, 10, 7])
    object_type = tf.reshape(object_type, [-1, 1])
    trajectories, marginal_confidences, shared_output, embeddings = self.marginal_model(image, x, y, yaw, past_states, object_type = object_type, is_sdc = is_sdc, training = training)
    is_veh = tf.cast(object_type == 1, tf.float32)
    is_ped = tf.cast(object_type == 2, tf.float32)
    is_cyc = 1 - is_veh - is_ped # (2B, 1)
    # (B*2, 8, 80, 2), (B*2, 8), (B*2, 1536)
    trajectories = tf.reshape(trajectories, [-1, 2, self.K, 80, 2])
    if not self.joint_conf_prediction:
      marginal_confidences = tf.reshape(marginal_confidences, [-1, 2, self.K])
      final_conf = confidence_cartesian_product(marginal_confidences[:, 0], marginal_confidences[:, 1])
    else:
      if self.embedding_method == "large":
        shared_conf = self.shared_conf_ffn(tf.concat([shared_output, embeddings], axis=1), training=training) # (B*2, 300)
      else:
        shared_conf = self.shared_conf_ffn(shared_output, training=training) # (B*2, 300)
      vv_conf = self.vv_conf_fc(shared_conf, training=training) # (B*2, 64)
      vp_conf = self.vp_conf_fc(shared_conf, training=training)
      vc_conf = self.vc_conf_fc(shared_conf, training=training)
      vs_conf = self.vs_conf_fc(shared_conf, training=training)
      is_sdc = tf.reshape(is_sdc, [-1, 2])
      
      object_type = tf.reshape(object_type, [-1, 2])
      object_type = object_type[:, 0] * object_type[:, 1] # (B,)
      is_vv = tf.reshape(tf.cast(object_type == 1, tf.float32), [-1, 1]) # (B, 1)
      is_s = tf.reshape(tf.cast(tf.reduce_sum(is_sdc, axis = 1) > 0, tf.float32), [-1, 1])
      is_vs = is_vv * is_s
      is_vv = is_vv * (1 - is_s)
      is_vp = tf.reshape(tf.cast(object_type == 2, tf.float32), [-1, 1]) # (B, 1)
      is_vc = 1 - is_vv - is_vp - is_vs

      is_vv = tf.concat([is_vv, is_vv], 0)
      is_vp = tf.concat([is_vp, is_vp], 0)
      is_vc = tf.concat([is_vc, is_vc], 0)
      is_vs = tf.concat([is_vs, is_vs], 0)

      conf = is_vv*vv_conf + is_vp*vp_conf + is_vc*vc_conf + is_vs*vs_conf # (B*2, 256)
      conf = tf.reshape(conf, [-1, 2, self.K, self.K])

      conf_0 = conf[:, 0]
      conf_1 = conf[:, 1]
      conf_1 = tf.transpose(conf_1, [0, 2, 1])
      average_conf = (conf_0 + conf_1)/2  # (B, K, K)
      valid_mask = is_veh*self.veh_is_valid + is_ped*self.ped_is_valid + is_cyc*self.cyc_is_valid
      valid_mask = tf.reshape(valid_mask, [-1, 2, self.K])
      valid_mask = valid_mask[:, 0][:, :, tf.newaxis]*valid_mask[:, 1][:, tf.newaxis, :]
      average_conf = valid_mask*average_conf + (1-valid_mask)*-1000000000.
      average_conf = tf.reshape(average_conf, [-1, self.K**2]) # (B, 8, 8)
      final_conf = self.softmax(average_conf)

    predictions = cartesian_product(trajectories[:, 0], trajectories[:, 1]) # (B, 64, 2, 80, 2)
    return predictions, final_conf, shared_output

  def compile(self, optimizer, multi_path_loss, multi_modal_loss, loss_tracker, multi_modal_loss_tracker, motion_metrics, metrics_config):
    super().compile(optimizer = optimizer, loss=multi_path_loss, metrics = [motion_metrics])
    self.multi_path_loss = multi_path_loss
    self.multi_modal_loss = multi_modal_loss
    self.multi_modal_loss_tracker = multi_modal_loss_tracker
    self.conf_loss_tracker = tf.keras.metrics.Mean(name = 'conf_loss')
    self.motion_metrics = motion_metrics
    self.metrics_config = metrics_config
    self.loss_tracker = loss_tracker
    self.veh_K = len(multi_path_loss.veh_centroids)
    self.ped_K = len(multi_path_loss.ped_centroids)
    self.cyc_K = len(multi_path_loss.cyc_centroids)
    self.K = max(self.veh_K, self.ped_K, self.cyc_K)
    self.veh_is_valid = pad_to_shape(tf.ones([1, self.veh_K], dtype = tf.float32), [1, self.K])
    self.ped_is_valid = pad_to_shape(tf.ones([1, self.ped_K], dtype = tf.float32), [1, self.K])
    self.cyc_is_valid = pad_to_shape(tf.ones([1, self.cyc_K], dtype = tf.float32), [1, self.K])

  def train_step(self, inputs):
    with tf.GradientTape() as tape:
      # gt_future_states has shape (32, 2, 91, 7)
      # gt_future_is_valid has shape (32, 2, 91)
      # past_states has shape (32, 2, 10, 7)
      # object_type has shape (32, 2)
      # x has shape (32, 2, 1)
      # y has shape (32, 2, 1)
      # yaw has shape (32, 2, 1)
      # image_0 has shape (32, 224, 448, 3)
      # image_1 has shape (32, 224, 448, 3)
      # scenario_id has shape (32, 1)
      # object_id has shape (32, 2)
      gt_trajectory = inputs['gt_future_states']
      gt_is_valid = inputs['gt_future_is_valid']
      prediction_start = self.metrics_config.track_history_samples + 1
      gt_targets = gt_trajectory[:, :, prediction_start:, :2] # (B, 2, 80, 2)
      weights = tf.cast(gt_is_valid[:, :, prediction_start:], tf.float32) # (B, 2, 80)

      images = tf.stack([inputs['image_0'], inputs['image_1']], axis = 1)
      x=inputs['x']
      y=inputs['y']
      yaw=inputs['yaw']
      past_states = inputs['past_states']
      object_type = inputs['object_type']
      pred_trajectory, confidences, _ = self(images, x=inputs['x'], y=inputs['y'], yaw=inputs['yaw'], past_states = inputs['past_states'], object_type = object_type, is_sdc = inputs['is_sdc'], training=True)
      loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, confidences, x, y, yaw, object_type)
    self.loss_tracker.update_state(loss)
    self.conf_loss_tracker.update_state(conf_loss)
    mm_loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences)
    self.multi_modal_loss_tracker.update_state(mm_loss)
    grads = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    object_type = tf.squeeze(inputs['object_type'])
    pred_score = confidences
    return {"loss": self.loss_tracker.result(), "mm_loss" : self.multi_modal_loss_tracker.result(), "conf_loss": self.conf_loss_tracker.result()}

  def test_step(self, inputs):
    gt_trajectory = inputs['gt_future_states']
    gt_is_valid = inputs['gt_future_is_valid']
    prediction_start = self.metrics_config.track_history_samples + 1
    gt_targets = gt_trajectory[:, :, prediction_start:, :2] # (B, 2, 80, 2)
    weights = tf.cast(gt_is_valid[:, :, prediction_start:], tf.float32) # (B, 2, 80)s
    
    images = tf.stack([inputs['image_0'], inputs['image_1']], axis=1)
    x = inputs['x']
    y = inputs['y']
    yaw = inputs['yaw']
    past_states = inputs['past_states']
    object_type = inputs['object_type']
    pred_trajectory, confidences, _ = self(images, x=x, y=y, yaw=yaw, past_states = past_states, object_type = object_type, is_sdc = inputs["is_sdc"], training=False)
    loss, conf_loss = self.multi_path_loss.call(gt_targets, weights, pred_trajectory, confidences, x, y, yaw, object_type)
    self.loss_tracker.update_state(loss)
    self.conf_loss_tracker.update_state(conf_loss)
    mm_loss = self.multi_modal_loss.call(gt_targets, weights, pred_trajectory, confidences)
    self.multi_modal_loss_tracker.update_state(mm_loss)

    object_type = inputs['object_type']
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
    images = tf.stack([inputs['image_0'], inputs['image_1']], axis=1)
    x = inputs['x']
    y = inputs['y']
    yaw = inputs['yaw']
    past_states = inputs['past_states']
    object_type = inputs['object_type']
    
    pred_trajectory, confidences, _ = self(images, x, y, yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training=False) # (B, 3, 80, 2)
    if truncate:
      interval = 5
      prediction_trajectory = pred_trajectory[:, :, :, (interval - 1)::interval]
      prediction_trajectory, confidences, _ = truncate_predictions(prediction_trajectory, confidences)
    else:
      prediction_trajectory = pred_trajectory
    return prediction_trajectory, confidences

  def inspect_step(self, inputs):
    """
      Parameters:
        inputs: A dictionary mapping from parsed feature names to batched tensors
      Returns: 
        trajectories: tensor of shape(B, 3, 16, 2)
        confidences: tensor of shape (B, 3)
    """
    images = tf.stack([inputs['image_0'], inputs['image_1']], axis=1)
    x = inputs['x']
    y = inputs['y']
    yaw = inputs['yaw']
    past_states = inputs['past_states']
    object_type = inputs['object_type']
    pred_trajectory, confidences = self.predict_step(inputs, truncate=False) # (B, 3, 80, 2)
    prediction_trajectory, confidences, _ = truncate_predictions(pred_trajectory, confidences, k=1) # (B, 1, 2, 80, 2), (B, 1)
    x = x.numpy()
    y = y.numpy()
    yaw = yaw.numpy()
    object_type = object_type.numpy()

    inspection_image = images[:, 0].numpy()
    prediction_trajectory = prediction_trajectory.numpy()
    for j in range(len(object_type)):
      if object_type[j, 0]!=3 and object_type[j, 1]!=3:
        continue
      world_to_image = transform_matrix(x[j, 0, 0], y[j, 0, 0], yaw[j, 0, 0])
      draw_trajectory(inspection_image[j], world_to_image, trajectory = prediction_trajectory[j, :, 0], show_image=False, colors=[(0, 255, 0)])
      draw_trajectory(inspection_image[j], world_to_image, trajectory = prediction_trajectory[j, :, 1], show_image=True, colors=[(255, 0, 0)])
    return prediction_trajectory, confidences

  @property
  def metrics(self):
    return [self.loss_tracker, self.multi_modal_loss_tracker, self.conf_loss_tracker]