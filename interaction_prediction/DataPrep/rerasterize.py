# This is the interaction prediction version!!!
from MotionPrediction.interaction_prediction.dataset import _parse_no_swap
from MotionPrediction.utils import truncate_predictions, transform_matrix, transform_points
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

CV2_SHIFT = 8
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT

def rerasterize_interaction(model, example):
  """
    Parameters:
      model: MotionPrediction Model: MOEModelWithVariableClusters
      example: one raw tf.example
    Returns:
      new_images: nparray of shape (2, 224, 448, 3)
  """
  inputs = _parse_no_swap(example.SerializeToString())
  
  for name in ["gt_future_states", "gt_future_is_valid", "past_states", "x", "y", "yaw", "is_sdc", "object_type"]:
    inputs[name] = inputs[name][:, tf.newaxis, ...]
    # print(name, inputs[name].shape)
    # gt_future_states (8, 1, 91, 7)
    # gt_future_is_valid (8, 1, 91)
    # past_states (8, 1, 10, 7)
    # x (8, 1, 1)
    # y (8, 1, 1)
    # yaw (8, 1, 1)
  inputs["image"] = tf.stack([inputs["image_0"], inputs["image_1"]], axis=0) # (2, 224, 448, 3)
  images = inputs["image"].numpy()
  past_states = inputs['past_states'] # (B, 10, 2)
  x = tf.squeeze(inputs['x'], axis = [1])
  y = tf.squeeze(inputs['y'], axis = [1])
  yaw = tf.squeeze(inputs['yaw'], axis = [1])
  object_type = inputs['object_type']
  pred_trajectory, confidences, _, _ = model(inputs['image'], x, y, yaw, past_states = past_states, object_type = object_type, is_sdc = inputs['is_sdc'], training = False) # (B, 3, 80, 2)
  interval = 5
  trajectories = pred_trajectory[:, :, (interval - 1)::interval]
  truncated_trajectories, truncated_confidences, _ = truncate_predictions(trajectories, confidences, k=1)
  num_tracks = 2
  new_images = []
  for i in range(num_tracks):
    image = np.zeros((224, 448, 3))
    x = inputs["x"][i, 0, 0]
    y = inputs["y"][i, 0, 0]
    yaw = inputs["yaw"][i, 0, 0]
    other_trajectories = np.concatenate([truncated_trajectories[:i, 0], truncated_trajectories[i+1:num_tracks, 0]], axis=0)
    world_to_image = transform_matrix(x, y, yaw)
    other_trajectories = other_trajectories.reshape([-1, 2])
    other_trajectories = transform_points(other_trajectories, world_to_image)
    other_trajectories = other_trajectories.reshape([-1, 16, 2])*CV2_SHIFT_VALUE
    for step in range(14, -1, -1):
      image *= 0.9
      current_points = other_trajectories[:, step:step+2] # (5, 2, 2)
      cv2.polylines(image, current_points.astype(np.int32), False, (255, 155, 55), thickness = 2, lineType = cv2.LINE_AA, shift = CV2_SHIFT)
    current_image = images[i]
    current_image[image > 0] = image[image > 0]
    new_images.append(current_image.astype(np.uint8))
  new_images = np.stack(new_images, axis=0)
  return new_images, pred_trajectory, confidences