##### InteractionPrediction!!!
import tensorflow as tf
import numpy as np
import yaml
import pprint
from waymo_open_dataset.metrics.python import config_util_py as config_util
from ..utils import truncate_predictions, transform_matrix, draw_trajectory
from ..utils import DotDict, load_config_data, LRRecorder
from .loss import JointPredMultiPathLoss, JointPredMultiModalLoss
from .train import load_model
from .metrics import MotionMetrics, _default_metrics_config
from .dataset import get_dataset, get_cyclist_dataset, get_ped_dataset, get_veh_dataset

class EnsembleModel(tf.keras.Model):
  def __init__(self, models, num_modes, dropout=0.2):
    super().__init__()
    self.models = models
    for model in models:
      model.trainable = False
    self.num_modes = num_modes
    self.metrics_config = _default_metrics_config()
    self.motion_metrics = MotionMetrics(self.metrics_config)

  def predict_step(self, batch, truncate=True):
    final_predictions = 0
    final_confidences = 0
    for model in self.models:
      predictions, confidences = model.predict_step(batch, truncate=False)
      predictions = predictions[:, :self.num_modes] # (B, K, 80, 2)
      confidences = confidences[:, :self.num_modes]
      final_predictions += predictions
      final_confidences += confidences
    final_predictions /= len(self.models)
    final_confidences /= len(self.models)
    if truncate:
      interval = 5
      final_predictions = final_predictions[:, :, :, (interval - 1)::interval]
      final_predictions, final_confidences, _ = truncate_predictions(final_predictions, final_confidences, k=6)
    return final_predictions, final_confidences  

  def prev_test_step(self, inputs):
    final_predictions, final_confidences = self.predict_step(inputs, truncate=False)
    final_predictions, final_confidences, _ = truncate_predictions(final_predictions, final_confidences, k=6)

    gt_trajectory = inputs['gt_future_states']
    gt_is_valid = inputs['gt_future_is_valid']

    object_type = inputs['object_type']
    self.motion_metrics.update_state(
        final_predictions,
        final_confidences,
        gt_trajectory,
        gt_is_valid,
        object_type)

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