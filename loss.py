import tensorflow as tf
import numpy as np

class MultiModalLoss(tf.keras.losses.Loss):
  def __init__(self, num_future_steps=80, name = 'multi_modal_loss'):
    super().__init__(name = name)
    self.num_future_steps = num_future_steps
  
  def call(self, gt, avails, pred, confidences, object_type = None, reduce_mean=True):
    """
    Call Arguments:
      targets: (B, 80, 2)
      pred: (B, 3, 80, 2)
      confidences: (B, 3)
      target_availabilities: (B, 80)
    
    Returns:
      loss:
        (1,) if reduce_mean
        (B, 1) otherwise
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    gt = tf.reshape(gt, [-1, 1, self.num_future_steps, 2])
    avails = tf.reshape(avails, [-1, 1, self.num_future_steps, 1])

    error = tf.reduce_sum(((gt - pred)*avails)**2, axis = [2,3]) # Outputs:(B, 3)
    
    error = tf.math.log(confidences + 1e-16) - 0.5 * error

    error = -tf.math.reduce_logsumexp(error, axis = 1, keepdims = True) # (B, 1)

    error /= self.num_future_steps

    if reduce_mean:
      return tf.reduce_mean(error)
    else:
      return error

class MTMPLoss(tf.keras.losses.Loss):
  def __init__(self, veh_centroids, ped_centroids, cyc_centroids, cls_weight, reg_weight, ped_weight=1., cyc_weight=1., num_future_steps = 80, name = 'mtmp_loss'):
    super().__init__(name = name)
    self.veh_centroids = tf.convert_to_tensor(veh_centroids)
    self.ped_centroids = tf.convert_to_tensor(ped_centroids)
    self.cyc_centroids = tf.convert_to_tensor(cyc_centroids)
    self.cls_weight = cls_weight
    self.reg_weight = reg_weight
    self.ped_weight = ped_weight
    self.cyc_weight = cyc_weight
    self.num_future_steps = num_future_steps
    self.K = max(len(veh_centroids), len(ped_centroids), len(cyc_centroids))

  def call(self, gt, avails, pred, confidences, x, y, yaw, object_type, reduce_mean = True):
    """
    Call Arguments:
      targets: (B, 80, 2)
      pred: (B, K, 80, 2)
      confidences: (B, K)
      target_availabilities: (B, 80)
      x: (B, 1)
      y: (B, 1)
      yaw: (B, 1)
    
    Returns:
      loss:
        (1,) if reduce_mean
        (B, 1) otherwise
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    
    c = tf.math.cos(yaw) # (B, 1)
    s = tf.math.sin(yaw) # (B, 1)
    x_hat = gt[:, :, 0] - x # (B, 80)
    y_hat = gt[:, :, 1] - y # (B, 80)
    gt_ego_x = c * x_hat + s * y_hat # (B, 80)
    gt_ego_y = -s * x_hat + c * y_hat # (B, 80)

    gt = tf.reshape(gt, [-1, 1, self.num_future_steps, 2])
    avails = tf.reshape(avails, [-1, 1, self.num_future_steps, 1])
    error = tf.reduce_sum(((gt - pred)*avails)**2, axis = [2,3]) # Outputs:(B, K)
    error = -self.cls_weight*tf.math.log(confidences + 1e-16) + self.reg_weight * 0.5 * error
    conf_error = -self.cls_weight*tf.math.log(confidences + 1e-16)

    gt_ego = tf.stack([gt_ego_x, gt_ego_y], axis = -1) # (B, 80, 2)
    # self.centroids has shape (K, 160)
    gt_ego = tf.reshape(gt_ego, [-1, 1, self.num_future_steps, 2])

    is_veh = object_type == 1 # object_type[:, 1] # (B, 1) object_type == 1
    is_ped = object_type == 2 # (B, 1)
    is_cyc = tf.math.logical_not(tf.math.logical_or(is_veh, is_ped))
    object_type_mask = tf.concat([is_veh, is_ped, is_cyc], axis=-1) # (B, 3)

    object_type_weight = tf.squeeze((tf.cast(is_veh, tf.float32) + tf.cast(is_ped, tf.float32)*self.ped_weight + tf.cast(is_cyc, tf.float32)*self.cyc_weight), axis=1) # (B,)

    veh_centroids = tf.reshape(self.veh_centroids, [1, -1, 80, 2])
    distance = ((gt_ego[:, :, -80:] - veh_centroids)**2)*avails[:, :, -80:] # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    veh_assignments = tf.argmin(distance, axis = 1) # (B,)
    veh_assignments_mask = tf.one_hot(veh_assignments, self.K) # (B, K)

    ped_centroids = tf.reshape(self.ped_centroids, [1, -1, 80, 2])
    distance = ((gt_ego[:, :, -80:] - ped_centroids)**2)*avails[:, :, -80:] # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    ped_assignments = tf.argmin(distance, axis = 1) # (B,)
    ped_assignments_mask = tf.one_hot(ped_assignments, self.K) # (B, K)

    cyc_centroids = tf.reshape(self.cyc_centroids, [1, -1, 80, 2])
    distance = ((gt_ego[:, :, -80:] - cyc_centroids)**2)*avails[:, :, -80:] # (B, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B, K)
    cyc_assignments = tf.argmin(distance, axis = 1) # (B,)
    cyc_assignments_mask = tf.one_hot(cyc_assignments, self.K) # (B, K)

    assignments_mask = tf.stack([veh_assignments_mask, ped_assignments_mask, cyc_assignments_mask], axis=1)
    assignments_mask = tf.boolean_mask(assignments_mask, object_type_mask) # (B, K)

    error = tf.cast(assignments_mask, tf.float32) * tf.cast(error, tf.float32) # (B, K)
    error = tf.reduce_sum(error, axis = 1)

    error *= object_type_weight
    error /= self.num_future_steps

    conf_error = tf.cast(assignments_mask, tf.float32) * tf.cast(conf_error, tf.float32) # (B, K)
    conf_error = tf.reduce_sum(conf_error, axis = 1)
    conf_error *= object_type_weight
    conf_error /= self.num_future_steps

    if reduce_mean:
      return tf.reduce_mean(error), tf.reduce_mean(conf_error)
    else:
      return error, conf_error, assignments_mask