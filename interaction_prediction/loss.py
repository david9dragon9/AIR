# This is the loss for interaction prediction
import tensorflow as tf
import numpy as np

class JointPredMultiModalLoss(tf.keras.losses.Loss):
  def __init__(self, name = 'jpmm_loss'):
    super().__init__(name = name)
  
  def call(self, gt, avails, pred, confidences, reduce_mean=True):
    """
    Call Arguments:
      targets: (B, 2, 80, 2)
      pred: (B, 64, 2, 80, 2)
      confidences: (B, 64)
      target_availabilities: (B, 2, 80)
    
    Returns:
      loss:
        (1,) if reduce_mean
        (B, 1) otherwise
    """
    assert len(pred.shape) == 5, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    gt = tf.reshape(gt, [-1, 1, 2, 80, 2])
    avails = tf.reshape(avails, [-1, 1, 2, 80, 1])

    error = tf.reduce_sum(((gt - pred)*avails)**2, axis = [2, 3,4]) # Outputs:(B, 64)
    
    error = tf.math.log(confidences + 1e-16) - 0.5 * error

    error = -tf.math.reduce_logsumexp(error, axis = 1, keepdims = True) # (B, 1)

    error /= 80.

    if reduce_mean:
      return tf.reduce_mean(error)
    else:
      return error

class JointPredMultiPathLoss(tf.keras.losses.Loss):
  def __init__(self, veh_centroids, ped_centroids, cyc_centroids, veh_weight = 1., ped_weight = 1., cyc_weight=1., cls_weight = 1., marginal_loss_weight=0., name = 'jpmp_loss'):
    super().__init__(name = name)
    self.veh_centroids = tf.convert_to_tensor(veh_centroids)
    self.ped_centroids = tf.convert_to_tensor(ped_centroids)
    self.cyc_centroids = tf.convert_to_tensor(cyc_centroids)
    self.K = len(self.veh_centroids)
    self.veh_weight = veh_weight
    self.ped_weight = ped_weight
    self.cyc_weight = cyc_weight
    self.cls_weight = cls_weight
    self.marginal_loss_weight = marginal_loss_weight

  def call(self, gt, avails, pred, confidences, x, y, yaw, object_type, reduce_mean = True):
    """
    Call Arguments:
      gt: (B, 2, 80, 2)
      pred: (B, K*K, 2, 80, 2)
      confidences: (B, K*K)
      avails: (B, 2, 80)
      x: (B, 2, 1)
      y: (B, 2, 1)
      yaw: (B, 2, 1)
      object_type: (B, 2)
    
    Returns:
      loss:
        (1,) if reduce_mean
        (B, 1) otherwise
    """
    assert len(pred.shape) == 5, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    x = tf.reshape(x, [-1, 1])
    y = tf.reshape(y, [-1, 1])
    yaw = tf.reshape(yaw, [-1, 1])
    gt = tf.reshape(gt, [-1, 80, 2])
    c = tf.math.cos(yaw) # (B*2, 1)
    s = tf.math.sin(yaw) # (B*2, 1)
    x_hat = gt[:, :, 0] - x # (B*2, 80)
    y_hat = gt[:, :, 1] - y # (B*2, 80)
    gt_ego_x = c * x_hat + s * y_hat # (B*2, 80)
    gt_ego_y = -s * x_hat + c * y_hat # (B*2, 80)

    gt = tf.reshape(gt, [-1, 1, 2, 80, 2])
    avails = tf.reshape(avails, [-1, 1, 2, 80, 1])
    error = tf.reduce_sum(((gt - pred)*avails)**2, axis = [2, 3, 4]) # Outputs:(B, 64)
    error = -self.cls_weight*tf.math.log(confidences + 1e-16) + 0.5 * error # (B, K*K)
    conf_error = -self.cls_weight*tf.math.log(confidences + 1e-16) # (B, K*K)

    gt_ego = tf.stack([gt_ego_x, gt_ego_y], axis = -1) # (B*2, 80, 2)
    # self.centroids has shape (K, 160)
    gt_ego = tf.reshape(gt_ego, [-1, 1, 80, 2]) # (B*2, 1, 80, 2)
    avails = tf.reshape(avails, [-1, 1, 80, 1]) # (B*2, 1, 80, 1)

    veh_centroids = tf.reshape(self.veh_centroids, [1, -1, 80, 2]) # (1, K, 80, 2)
    distance = ((gt_ego - veh_centroids)**2)*avails # (B*2, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B*2, K)
    veh_assignments = tf.argmin(distance, axis = 1) # (B*2,)
    veh_assignments = tf.cast(veh_assignments, tf.int32)

    ped_centroids = tf.reshape(self.ped_centroids, [1, -1, 80, 2]) # (1, K, 80, 2)
    distance = ((gt_ego - ped_centroids)**2)*avails # (B*2, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B*2, K)
    ped_assignments = tf.argmin(distance, axis = 1) # (B*2,)
    ped_assignments = tf.cast(ped_assignments, tf.int32)

    cyc_centroids = tf.reshape(self.cyc_centroids, [1, -1, 80, 2]) # (1, K, 80, 2)
    distance = ((gt_ego - cyc_centroids)**2)*avails # (B*2, K, 80, 2)
    distance = tf.reduce_sum(distance, [2,3]) # (B*2, K)
    cyc_assignments = tf.argmin(distance, axis = 1) # (B*2,)
    cyc_assignments = tf.cast(cyc_assignments, tf.int32)

    object_type_mask = tf.math.reduce_max(object_type, axis=1)
    object_type_mask = self.veh_weight*tf.cast(object_type_mask==1, tf.float32) +\
                       self.ped_weight*tf.cast(object_type_mask==2, tf.float32) +\
                       self.cyc_weight*tf.cast(object_type_mask==3, tf.float32)

    object_type_mask = tf.reshape(object_type_mask, [-1, 1])

    object_type = tf.reshape(object_type, [-1,])
    is_veh = tf.cast(object_type == 1, tf.int32) # (2*B,)
    is_ped = tf.cast(object_type == 2, tf.int32)
    is_cyc = 1 - is_veh - is_ped

    assignments = is_veh*veh_assignments + is_ped*ped_assignments + is_cyc*cyc_assignments # (2*B,)
    
    assignments = tf.reshape(assignments, [-1, 2])
    assignments_0 = assignments[:, 0] # (B,)
    assignments_1 = assignments[:, 1] # (B,)

    marginal_0 = tf.reduce_sum(tf.reshape(confidences, [-1, self.K, self.K]), axis=2) # (B, K)
    marginal_0 = -tf.math.log(marginal_0 + 1e-16) # (B,)
    marginal_0 = tf.one_hot(assignments_0, depth=self.K)*marginal_0

    marginal_1 = tf.reduce_sum(tf.reshape(confidences, [-1, self.K, self.K]), axis=1)
    marginal_1 = -tf.math.log(marginal_1 + 1e-16) # (B,)
    marginal_1 = tf.one_hot(assignments_1, depth=self.K)*marginal_1

    marginal_error = tf.reduce_sum(marginal_0, axis=1) + tf.reduce_sum(marginal_1, axis=1)

    assignments = self.K*assignments_0 + assignments_1 # (B,)
    assignments_mask = tf.one_hot(assignments, self.K**2) # (B, 64)
    
    error = assignments_mask * error # (B, K**2)
    error = error * object_type_mask
    error = tf.reduce_sum(error, axis = 1) + marginal_error * self.marginal_loss_weight
    error /= 80.

    conf_error = assignments_mask * conf_error
    conf_error = conf_error * object_type_mask
    conf_error = tf.reduce_sum(conf_error, axis = 1) + marginal_error * self.marginal_loss_weight
    conf_error /= 80.

    if reduce_mean:
      return tf.reduce_mean(error), tf.reduce_mean(conf_error)
    else:
      return error, conf_error