import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow_graphics.math.interpolation import bspline

def get_trajectories(dataset):
  trajectories = []
  avails = []
  object_types = []

  for i, batch in enumerate(dataset):
    future_states = tf.squeeze(batch['gt_future_states'], axis = 1)[:, 11:, :2]
    future_is_valid = tf.squeeze(batch['gt_future_is_valid'], axis = 1)[:, 11:]
    x = batch['x']
    y = batch['y']
    yaw = batch['yaw']
    x = tf.squeeze(x, axis = 1)
    y = tf.squeeze(y, axis = 1)
    yaw = tf.squeeze(yaw, axis = 1)
    c = tf.math.cos(yaw)
    s = tf.math.sin(yaw)
    object_type = tf.squeeze(batch['object_type'], axis = 1)
    
    future_x = future_states[:, :, 0] # (B, 80)
    future_y = future_states[:, :, 1] # (B, 80)
    future_x_hat = future_x - x # (B, 80)
    future_y_hat = future_y - y # (B, 80)
    future_ego_x = c * future_x_hat + s * future_y_hat # (B, 80)
    future_ego_y = -s * future_x_hat + c * future_y_hat # (B, 80)
    future_states = tf.stack([future_ego_x, future_ego_y], axis = -1)
    trajectories.append(future_states)
    avails.append(future_is_valid)
    object_types.append(object_type)
    if i % 1000 == 0:
      print(i)
    
  trajectories = tf.concat(trajectories, axis = 0)
  avails = tf.concat(avails, axis = 0)
  object_types = tf.concat(object_types, axis = 0)
  trajectories = trajectories.numpy()
  avails = avails.numpy()
  object_types = object_types.numpy()
  np.save("drive/MyDrive/Motion/trajectories.npy", trajectories)
  np.save("drive/MyDrive/Motion/avails.npy", avails)
  np.save("drive/MyDrive/Motion/object_types.npy", object_types)
  return trajectories, avails, object_types

def cluster(trajectories, avails, K = 8, num_iters = 30):
  num = trajectories.shape[1]
  trajectories = trajectories.copy().reshape([-1, 2*num])
  avails = avails.reshape([-1, num, 1])
  avails = np.concatenate((avails, avails), axis = 2)
  avails = avails.reshape([-1, 2*num])
  centroids = trajectories.copy()[0:K*17:17,:] # (8, 160)
  for iteration in range(num_iters):
    assignments = m_step(trajectories, avails, centroids)
    e_step(trajectories, avails, centroids, assignments)
  return assignments, centroids, trajectories, avails

def chunked_cluster(trajectories, avails, initial_centroids = None, K = 8, num_iters = 30, chunk_size=250000):
  num = int(trajectories.shape[1])
  trajectories = trajectories.copy().reshape([-1, 2*num])
  avails = avails.reshape([-1, num, 1])
  avails = np.concatenate((avails, avails), axis = 2)
  avails = avails.reshape([-1, 2*num])
  if initial_centroids is not None:
    centroids = initial_centroids.copy()
  else:
    centroids = trajectories.copy()[0:K*17:17,:] # (8, 160)
  N = len(trajectories)
  for iteration in range(num_iters):
    print(iteration)
    assignments_list = []
    for i in range(0, N, chunk_size):
      j = min(i + chunk_size, N)
      assignments_list.append(m_step(trajectories[i:j], avails[i:j], centroids))
    assignments = np.concatenate(assignments_list, axis = 0)
    e_step(trajectories, avails, centroids, assignments)
  return assignments, centroids, trajectories, avails

def m_step(trajectories, avails, centroids):
  """
    Parameters:
      trajectories: nparray of shape (B, 160)
      avails: nparray of shape (B, 160)
      centroids: nparray of shape (8, 160)
    
    Returns:
      assignments: nparray of shape(B,)(Each trajectory has an assignment to a cluster)
  """
  K = len(centroids)
  num = trajectories.shape[1]//2
  assert num != 160, "num is 160"
  a = trajectories.reshape([-1, 1, 2*num])
  b = centroids.reshape([1, K, 2*num])
  reshaped_avails = avails.reshape([-1, 1, 2*num])
  distance = ((a-b)**2)*reshaped_avails # (B, 8, 160)
  distance = np.sum(distance, axis = 2) # (B, 8)
  assignments = np.argmin(distance, axis = 1) # (B,)
  print('total cost:', np.sum(np.min(distance, axis = 1).astype(np.float64)))
  return assignments

def e_step(trajectories, avails, centroids, assignments, K = 8):
  """
    Parameters:
      trajectories: nparray of shape (B, 160)
      avails: nparray of shape (B, 160)
      centroids: nparray of shape (8, 160)
      assignments: nparray of shape(B,)(Each trajectory has an assignment to a cluster)

    Returns:
      None: centroids are changed in place.
  """
  K = len(centroids)
  for i in range(K):
    members = np.where(assignments == i)
    member_trajectories = trajectories[members] # (C, 160)
    member_avails = avails[members] # (C, 160)
    sum_trajectory = np.sum(member_trajectories*member_avails, axis = 0) # (160,)
    sum_avails = np.sum(member_avails, axis = 0) + 1e-6 # (160,)
    centroids[i] = sum_trajectory/sum_avails

def visualize_clusters(assignments, centroids, avails):
  colors = [(255, 0, 0),
          (255, 255, 0),
          (255, 255, 255),
          (0,255, 255),
          (0,255,0),
          (0,0,255),
          (255,0,255),
          (255, 255, 100)]
  for i in range(len(centroids)):
    indices = np.where(assignments == i)[0]
    centroid_avails = np.any(avails[indices], axis = 0)
    print(f"the {i}th cluster has this many members:{len(indices)}")
    trajectory = centroids[i][centroid_avails].reshape([-1, 2]).astype(np.int64)*2 + 112
    image = np.zeros((224, 448, 3))
    cv2.polylines(image, [trajectory], False, color = colors[i%8])    
    plt.imshow(image/255)
    plt.show()

def visualize_trajectories(trajectories, avails, indices): # trajectories has shape (B, 160)
  colors = [(255, 0, 0),
          (255, 255, 0),
          (255, 255, 255),
          (0,255, 255),
          (0,255,0),
          (0,0,255),
          (255,0,255),
          (255, 255, 100)]
  image = np.zeros((224,448,3))
  for index in indices:
    track_trajectory = trajectories[index]
    track_avail = avails[index]
    track_trajectory = track_trajectory[track_avail]
    track_trajectory = 2*track_trajectory.reshape([1,-1,2]).astype(np.int64) + 112
    cv2.polylines(image, track_trajectory, False, color = colors[index % 8])
  plt.imshow(image/255)
  plt.show()

def inspect_trajectory(dataset, index, batch_size = 32):
  batch_index = index//batch_size
  index_within_batch = index % batch_size
  for i, batch in enumerate(dataset):
    if i < batch_index:
      continue
    image = batch['image'][index_within_batch]
    future_states = tf.squeeze(batch['gt_future_states'], axis = 1)[:, 11:, :2]
    future_is_valid = tf.squeeze(batch['gt_future_is_valid'], axis = 1)[:, 11:] # (B, 80)
    x = batch['x']
    y = batch['y']
    yaw = batch['yaw']
    x = tf.squeeze(x, axis = 1)
    y = tf.squeeze(y, axis = 1)
    yaw = tf.squeeze(yaw, axis = 1)
    c = tf.math.cos(yaw)
    s = tf.math.sin(yaw)
    
    future_x = future_states[:, :, 0] # (B, 80)
    future_y = future_states[:, :, 1] # (B, 80)
    future_x_hat = future_x - x # (B, 80)
    future_y_hat = future_y - y # (B, 80)
    future_ego_x = c * future_x_hat + s * future_y_hat # (B, 80)
    future_ego_y = -s * future_x_hat + c * future_y_hat # (B, 80)
    future_states = tf.stack([future_ego_x, future_ego_y], axis = -1) # (B, 80, 2)
    trajectory = future_states[index_within_batch].numpy()
    avails = future_is_valid[index_within_batch].numpy()
    trajectory = (2.5*trajectory[avails]).astype(np.int64) + 112
    image = image.numpy()
    image = np.zeros((224,448,3))
    cv2.polylines(image, [trajectory], False, color = (0,255,0))
    plt.imshow(image/255)
    plt.show()
    break

def smooth(trajectories, avails, centroids, assignments):
  """
  Arguments:
    trajectories: nparray of shape (X, 80, 2)
    avails: nparray of shape (X, 80)
    centroids: nparray of shape (n, 160)
    assignments: nparray of shape (X,)
  
  Returns:
    new_centroids: nparray of shape (n, 160)
  """
  n = len(centroids)
  new_centroids = np.zeros((n, 160))
  histories = []
  for i in range(n):
    print(i)
    initial_knots = tf.convert_to_tensor(centroids[i].reshape([80, 2])[9::10])
    model = get_cluster_model(initial_knots)
    opt = tf.keras.optimizers.SGD(learning_rate=10)
    model.compile(opt, loss=cluster_loss)
    current_trajectories = trajectories[assignments==i]
    current_avails = avails[assignments==i]
    output = np.stack([current_trajectories, np.stack([current_avails, current_avails], axis=-1)], axis=1)
    num_examples = len(current_trajectories)
    history = model.fit(x = np.zeros((num_examples,)), y=output, batch_size=num_examples, epochs=100, verbose=0)
    histories.append(history)
    print("loss", history.history["loss"][-1])
    current_centroid = model(np.array([0])).numpy()
    new_centroids[i] = current_centroid.reshape([160,])
  visualize_centroids(new_centroids)
  return new_centroids

def cluster_and_get_all_avails(filtered_trajectories, filtered_avails, K, num_iters, chunk_size):
  assignments_K, centroids_K, trajectories_K, avails_K = chunked_cluster(filtered_trajectories, filtered_avails, K=K, num_iters=num_iters, chunk_size=chunk_size)
  np.save("drive/MyDrive/Motion/clusters/filtered_veh_64.npy", centroids_K)
  np.save("drive/MyDrive/Motion/clusters/filtered_assignments_64.npy", assignments_K)
  all_avails_K = []
  for i in range(K):
    all_avails_K.append(avails_K[np.where(assignments_K==i)])
  return assignments_K, centroids_K, trajectories_K, avails_K, all_avails

def visualize_centroids(centroids, all_avails=None):
  """
    Call Arguments:
      centroids: (K, 160)
      all_avails: python list of nparrays of shape (B, 80)
  """
  K = len(centroids)
  num = centroids.shape[1]//2
  for i in range(K):
    if all_avails != None:
      avails = np.any(all_avails[i], axis=0)
      centroid = (2.5*centroids[i].reshape([num, 2])[avails]).astype(np.int32) + 112
    else:
      centroid = (2.5*centroids[i].reshape([num, 2])).astype(np.int32) + 112
    print(i)
    image = np.zeros((224, 448, 3))
    cv2.polylines(image, [centroid], False, (255, 255, 255))
    for pt in centroid[::4]:
      cv2.circle(image, (pt[0], pt[1]), 1, (255, 0, 0))
    plt.figure(figsize=(10, 20))
    plt.imshow(image/255)
    plt.show()

def chunked_m_step(trajectories, avails, centroids, chunk_size=125000):
  N = len(trajectories)
  trajectories = trajectories.copy().reshape([-1, 160])
  avails = avails.reshape([-1, 80, 1])
  avails = np.concatenate((avails, avails), axis = 2)
  avails = avails.reshape([-1, 160])
  assignments_list = []
  for i in range(0, N, chunk_size):
    j = min(i + chunk_size, N)
    assignments_list.append(m_step(trajectories[i:j], avails[i:j], centroids))
  assignments = np.concatenate(assignments_list, axis = 0)
  return assignments

def get_cluster_model(initial_knots):
    """
    initial_knots: tensor of shape (8, 2)
    """
    dummy_input = tf.keras.layers.Input(shape = (1,)) 
    knots = tf.keras.layers.Dense(16)(dummy_input)
    knots = tf.reshape(knots, (-1, 1, 2, 8)) # (B, 1, 2, 8)
    initial_knots = initial_knots[tf.newaxis, tf.newaxis, :, :]
    knots = knots + tf.transpose(initial_knots, [0, 1, 3, 2])
    max_pos = 8 - 3
    positions = tf.expand_dims(tf.range(start = 0.0, limit = max_pos, delta = max_pos/80, dtype= knots.dtype), axis = -1)
    spline = bspline.interpolate(knots, positions, 3, False)
    spline = tf.squeeze(spline, axis = 1)
    pred = tf.transpose(spline, perm = [1,2,0,3]) # (B, K, 80, 2)
    pred = tf.reshape(pred, [-1, 80, 2])
    model = tf.keras.Model(inputs=[dummy_input], outputs =[pred])
    return model

def cluster_loss(y_true, y_pred):
  return tf.reduce_mean(((y_true[:, 0] - y_pred)**2) * y_true[:, 1])

def show_trajectory(trajectory):
  """
  trajectory: of shape (80, 2) or (160) or (1, 80, 2)
  """
  image = np.zeros((224, 448, 3))
  pts = (2.5*trajectory.reshape([80, 2])).astype(np.int32) + 112
  cv2.polylines(image, pts, False, (255, 255, 255))
  for pt in pts:
    cv2.circle(image, (pt[0], pt[1]), 1, (255, 0, 0))
  plt.figure(figsize=(10, 20))
  plt.imshow(image)
  plt.show()