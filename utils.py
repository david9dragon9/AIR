import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import yaml
from tensorflow.keras.callbacks import Callback

CV2_SHIFT = 8
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT

class WorldToEgo(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
  
  def call(self, coords, x, y, yaw):
    """
    Call Args:
      coords: (B, N, 2)
      x: (B, 1)
      y: (B, 1)
      yaw: (B, 1)
    Returns:
      coords: (B, N, 2)
    """
    c = tf.math.cos(yaw)
    s = tf.math.sin(yaw)
    x_hat = coords[..., 0] - x # (B, N)
    y_hat = coords[..., 1] - y # (B, N)
    coords_ego_x = c * x_hat + s * y_hat # (B, N)
    coords_ego_y = -s * x_hat + c * y_hat # (B, N)
    coords = tf.stack([coords_ego_x, coords_ego_y], axis=-1)
    return coords

class EgoToWorld(tf.keras.layers.Layer):
  def __init__(self):
    super().__init__()
  
  def call(self, coords, x, y, yaw):
    """
    Call Args:
      coords: (B, N, 2)
      x: (B, 1)
      y: (B, 1)
      yaw: (B, 1)
    Returns:
      coords: (B, N, 2)
    """
    c = tf.math.cos(yaw)
    s = tf.math.sin(yaw)
    coords_x = c*coords[...,0] - s*coords[...,1] + x # (B, 80)
    coords_y = s*coords[...,0] + c*coords[...,1] + y # (B, 80)
    coords = tf.stack([coords_x, coords_y], axis=-1)
    return coords

class LRRecorder(Callback):
    """Record current learning rate. """
    def on_epoch_begin(self, epoch, logs=None):
      lr = self.model.optimizer._decayed_lr(tf.float32)
      print(f"The current learning rate is {lr.numpy()}")

def truncate_predictions(trajectories, confidences, k=6):
  """
    Parameters:
      trajectories: tensor of shape (B, K, 16, 2)
      confidences: tensor of shape (B, K)
    Returns:
      truncated_trajectories: (B, min(K,6), 16,2)
      truncated_confidences: (B, min(K,6))
  """
  if confidences.shape[1] <= k:
    return trajectories, confidences, None
  truncated_confidences, indices = tf.math.top_k(confidences, k=k)
  truncated_trajectories = tf.gather(trajectories, indices, batch_dims = 1)
  return truncated_trajectories, truncated_confidences, indices

def transform_points(points, world_to_image):
  """
  pts are nparray of shape(B, 2)
  world_to_image is nparray of shape(3,3)
  Returns nparray of shape(B, 2)
  """
  world_to_image = world_to_image.T
  return points @ world_to_image[:2,:2] + world_to_image[2,:2]

def transform_matrix(cx, cy, yaw):
  """
  Returns nparray of shape (3,3)
  """
  c = np.cos(yaw)
  s = np.sin(yaw)
  return np.array([[2.5*c, 2.5*s, -2.5*(c*cx + s*cy)+112],
                   [-2.5*s, 2.5*c, -2.5*(-s*cx + c*cy)+112],
                   [0., 0., 1.              ]])

def ego_to_world(cx, cy, yaw):
  """
  Returns nparray of shape(3,3)
  """
  c = np.cos(yaw)
  s = np.sin(yaw)
  return np.array([[c, -s, cx],
                   [s, c, cy],
                   [0, 0, 1]])


def draw_trajectory(img, world_to_image, trajectory = None, gt_trajectory=None, avails=None, show_image=True, colors=None):
  """
    img: (224,448,3)
    trajectory: np array of (16, 80, 2) of (x,y) in world coordinates
    world_to_image: 3x3 transform matrix to map from world to image coordinates
    Returns: None.  img is modified in place to show the trajectory
    gt_trajectory: np array of (80, 2)
  """
  if colors is None:
    colors = [
            (0, 0, 255),
            (255, 0, 0),
            (255, 255, 255),
            (0,255, 255),
            (255,155,123),
            (0,0,255),
            (255,0,255),
            (255, 255, 100)]
  num_modes = trajectory.shape[0] if trajectory is not None else 1
  if gt_trajectory is not None:

    gt_transformed = transform_points(gt_trajectory, world_to_image)
    pts = np.array([gt_transformed[29], gt_transformed[59], gt_transformed[79]]).astype(np.int32)
    for pt in pts:
      cv2.circle(img, (pt[0], pt[1]), 2, (0, 255, 0), -1)
    gt_transformed = gt_transformed*CV2_SHIFT_VALUE
    gt_transformed = gt_transformed.astype(np.int64)
    if avails is not None:
      gt_transformed = gt_transformed[avails]
    cv2.polylines(img, [gt_transformed], False, color = (0,255,0), thickness=1, lineType=cv2.LINE_AA, shift=CV2_SHIFT)


  if trajectory is not None:
    for i in range(num_modes):
      transformed = transform_points(trajectory[i, :, :], world_to_image)
      pts = np.array([transformed[29], transformed[59], transformed[79]]).astype(np.int32) 
      for pt in pts:
        cv2.circle(img, (pt[0], pt[1]), 2, colors[i%8], -1)

      transformed = transformed*CV2_SHIFT_VALUE
      transformed = transformed.astype(np.int64)
      cv2.polylines(img, [transformed], False, color = colors[i%8], thickness=1, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
  if show_image:
    plt.figure(figsize=(15,30))
    plt.imshow(img[::-1]/255)
    plt.show()

def load_cnn_model(name='efficient_net_b3', input_shape = (224, 448, 3)):
  efficient_net_b3 = tf.keras.applications.EfficientNetB3(
    include_top = False,
    input_shape = input_shape
  )
  for layer in efficient_net_b3.layers:
    layer.trainable = True

  return efficient_net_b3

# boxes: Shape(B, 5): nparray of [centroidx, centroidy, length, width, yaw]
# Returns: nparray of shape (B, 4, 2)
def get_corners_in_world_coordinates(boxes):
  """
  boxes: Shape(B, 5): nparray of [centroidx, centroidy, length, width, yaw]
  Returns: nparray of shape (B, 4, 2)
  """
  B, _ = boxes.shape
  result = np.zeros((B, 4, 2), dtype = float)
  cx = boxes[:, 0]
  cy = boxes[:, 1]
  half_w = boxes[:, 3]/2
  half_l = boxes[:, 2]/2
  yaw = boxes[:, 4]
  c = np.cos(yaw)
  s = np.sin(yaw)

  cl = c * half_l
  sw = s * half_w
  sl = s * half_l
  cw = c * half_w

  result[:, 0, 0] = cl - sw + cx 
  result[:, 1, 0] = cl + sw + cx
  result[:, 2, 0] = -cl + sw + cx 
  result[:, 3, 0] = -cl - sw + cx 
  result[:, 0, 1] = sl + cw + cy 
  result[:, 1, 1] = sl - cw + cy 
  result[:, 2, 1] = -sl - cw + cy
  result[:, 3, 1] = -sl + cw + cy
  return result

def road_segment_color(rs_type):
  rs_color = {1: (1,1,1), # LaneCenter-Freeway = 1
              2: (217/255, 221/255, 1), # LaneCenter-SurfaceStreet = 2
              3: (0,.5,1), # LaneCenter-BikeLane = 3
              6: (200/255,200/255,200/255), # RoadLine-BrokenSingleWhite = 6
              7: (1,1,1), # RoadLine-SolidSingleWhite = 7
              8: (.8,.8,.8), #  RoadLine-SolidDoubleWhite = 8
              9: (1,1,0), # RoadLine-BrokenSingleYellow = 9
              10: (.8,.8,0), # RoadLine-BrokenDoubleYellow = 10
              11: (.9,.9,0), #Roadline-SolidSingleYellow = 11, 
              12: (.7,.7,0), #Roadline-SolidDoubleYellow=12, 
              13: (.75,.75,0), #RoadLine-PassingDoubleYellow = 13,
              15: (.5,0,1), #RoadEdgeBoundary = 15, 
              16: (.5,0,1), #RoadEdgeMedian = 16, 
              17: (1,0,0), #StopSign = 17, 
              18: (0,0,1), #Crosswalk = 18, 
              19: (.6,.5,.6) #SpeedBump = 19
              }
  return rs_color[rs_type] if rs_type in rs_color else (.5,.5,.5)

class DotDict(dict):
    """dot.notation access to dictionary attributes

    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_config_data(experiment_name: str) -> dict:
    with open(f"drive/MyDrive/Motion/MotionPrediction/experiments/{experiment_name}") as f:
        cfg: dict = yaml.load(f, Loader=yaml.FullLoader)
    return DotDict(cfg)

def cartesian_product(a,b):
  """
  Note: this cartesian product only supports tiling of dimension 1(first dimension is batch)
  """
  length_a = a.shape[1]
  length_b = b.shape[1]
  a = tf.reshape(a, [-1, length_a, 1, 80, 2])
  b = tf.reshape(b, [-1, 1, length_b, 80, 2])
  a = tf.tile(a, [1, 1, length_b, 1, 1])
  b = tf.tile(b, [1, length_a, 1, 1, 1])
  a = tf.reshape(a, [-1, length_a*length_b, 80, 2])
  b = tf.reshape(b, [-1, length_a*length_b, 80, 2])
  c = tf.stack([a,b], 2)
  return c
  
def confidence_cartesian_product(a,b):
  length_a = a.shape[1]
  length_b = b.shape[1]
  a = tf.reshape(a, [-1, 1, length_a])
  b = tf.reshape(b, [-1, length_b, 1])
  a = tf.tile(a, [1, length_b, 1])
  b = tf.tile(b, [1, 1, length_a])
  a = tf.reshape(a, [-1, length_a*length_b])
  b = tf.reshape(b, [-1, length_a*length_b])
  c = a * b
  return c

def pad_to_shape(x, shape, pad_val=0):
  pad = shape - tf.minimum(tf.shape(x), shape)
  zeros = tf.zeros_like(pad)
  x = tf.pad(x, tf.stack([zeros, pad], axis=1), constant_values = pad_val)
  return tf.reshape(tf.slice(x, zeros, shape), shape)

def calculate_lr(steps, yaml_file):
  cfg = load_config_data(yaml_file)
  train_params = DotDict(cfg.train_params)
  model_params = DotDict(cfg.model_params)
  initial_lr = train_params.initial_lr
  decay_steps = train_params.steps
  alpha = train_params.alpha
  if steps < decay_steps:
    return initial_lr*(alpha + (1-alpha)*(decay_steps-steps)/decay_steps)
  else:
    return initial_lr*alpha