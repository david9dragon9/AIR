import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ..utils import get_corners_in_world_coordinates
from ..utils import transform_points
from ..utils import transform_matrix
from ..utils import road_segment_color

CV2_SHIFT = 8
CV2_SHIFT_VALUE = 2 ** CV2_SHIFT


def rasterize(parsed):
  """
  Parameters:
    parsed: a parsed example
  
  Returns:
    batch_images: a nparray of rasterized images of shape(B, 224,448, 3) dtype = float32
  """
  decoded_example = parsed

  past_states = tf.stack([
        decoded_example['state/past/x'],
        decoded_example['state/past/y'],
        decoded_example['state/past/length'],
        decoded_example['state/past/width'],
        decoded_example['state/past/bbox_yaw']
    ], -1)
  cur_states = tf.stack([
        decoded_example['state/current/x'],
        decoded_example['state/current/y'],
        decoded_example['state/current/length'],
        decoded_example['state/current/width'],
        decoded_example['state/current/bbox_yaw']
    ], -1)
  states = tf.concat([past_states, cur_states], axis = 1)
  past_is_valid = decoded_example['state/past/valid'] > 0
  current_is_valid = decoded_example['state/current/valid'] > 0
  is_valid = tf.concat([past_is_valid, current_is_valid], axis = 1)
  is_valid = tf.reduce_any(is_valid, 1)
  valid_states = tf.boolean_mask(states, is_valid)
  tracks_to_predict = parsed['state/tracks_to_predict']
  current_is_valid = tf.squeeze(current_is_valid, axis = 1)
  orig_to_valid_map = (tf.cumsum(tf.cast(is_valid, dtype = tf.int32)) - 1).numpy()
  tracks = tf.where(tracks_to_predict > 0)
  tracks = tracks.numpy().reshape(-1)
  current_is_valid = current_is_valid.numpy()

  r_valid_states = tf.transpose(valid_states, perm = [1,0,2]) # (11,58,5)
  r_valid_states = tf.reshape(r_valid_states, (-1,5))
  corners = get_corners_in_world_coordinates(r_valid_states) # (58*11, 4, 2)

  ego_info = {}
  current_x = parsed['state/current/x'].numpy().reshape(-1)
  current_y = parsed['state/current/y'].numpy().reshape(-1)
  current_yaw = parsed['state/current/bbox_yaw'].numpy().reshape(-1)

  # Prepare the road data
  xyz_road = parsed['roadgraph_samples/xyz']
  is_valid_road = parsed['roadgraph_samples/valid']
  road_type = parsed['roadgraph_samples/type']

  xy_road = xyz_road[:,:2]
  is_valid_road = tf.squeeze(is_valid_road)
  valid_xy_road = tf.boolean_mask(xy_road, is_valid_road)
  dir_road = parsed['roadgraph_samples/dir']
  dir_xy_road = dir_road[:, :2]
  valid_dir_xy_road = tf.boolean_mask(dir_xy_road, is_valid_road)
  valid_road_type = np.squeeze(tf.boolean_mask(road_type, is_valid_road).numpy())
  road_ids = np.squeeze(tf.boolean_mask(parsed['roadgraph_samples/id'], is_valid_road).numpy())

  valid_xy_plus_dir = valid_xy_road + valid_dir_xy_road
  valid_xy_plus_dir = valid_xy_plus_dir.numpy()
  valid_xy_road = valid_xy_road.numpy()

  tl_state = parsed['traffic_light_state/current/state']
  tl_ids = parsed['traffic_light_state/current/id']
  tl_valid = parsed['traffic_light_state/current/valid']
  valid_tl_states = tf.boolean_mask(tl_state, tl_valid).numpy()
  valid_tl_ids = tf.boolean_mask(tl_ids, tl_valid).numpy()

  batch_images = np.zeros((len(tracks), 224,448, 3), dtype=np.float32)
  for track_index, track in enumerate(tracks):
    if not current_is_valid[track]:
      print("WARNING! Found a track that is not valid in current frame!")
      batch_images[track_index] = None
      continue
    track_in_valid_index = orig_to_valid_map[track]

    cx = current_x[track]
    cy = current_y[track]
    yaw = current_yaw[track]
    # generate the transfer matrix
    transform = transform_matrix(cx, cy, yaw)
    transformed = transform_points(corners, transform)

    tl_colors = [(1,1,1), # white Unknown = 0
                (1,0,0), # red Arrow_Stop = 1
                (1,1,0), # yellow Arrow_Caution = 2
                (0,1,0), # green Arrow_go = 3
                (1,0,0), # red stop = 4
                (1,1,0), # yellow caution = 5
                (0,1,0), # green go = 6
                (1,115/255,0), # red flashing_stop = 7
                (212/255,1,0)] # yellow flashing caution = 8
    # Drawing the road
    road_img = np.zeros((224,448,3), dtype = np.float32)
    valid_xy_road_in_img = transform_points(valid_xy_road, transform)*CV2_SHIFT_VALUE
    valid_xy_plus_dir_in_img = transform_points(valid_xy_plus_dir, transform)*CV2_SHIFT_VALUE
    road_pts = np.stack([valid_xy_road_in_img, valid_xy_plus_dir_in_img], 1).astype(np.int64)
    for rs_type in [1,2,3,6,7,8,9,10,11,12,13,15,16,17,18,19]:
      type_indexes = np.where(valid_road_type == rs_type)
      cv2.polylines(road_img, road_pts[type_indexes], False, color = road_segment_color(rs_type), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    for i,tl_state in enumerate(valid_tl_states):
      lane_id = valid_tl_ids[i]
      tl_road_pt_indexes = np.where(road_ids == lane_id)[0]
      cv2.polylines(road_img, road_pts[tl_road_pt_indexes], False, tl_colors[tl_state], lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    road_img = np.clip(road_img, 0, 1)

    pts = np.reshape(transformed*CV2_SHIFT_VALUE, (11, -1, 4, 2)).astype(np.int64)
    out_img = np.zeros((224,448, 3), dtype = np.float32)
    for i in range(11):
      out_img *= 0.85
      cv2.fillPoly(out_img, pts[i], color = (1,1,0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
      # draw the ego in green
      cv2.fillPoly(out_img, pts[i][track_in_valid_index:track_in_valid_index+1], color = (0,1,0), lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    out_img = np.clip(out_img, 0, 1)

    # Combine road and car images
    road_img[out_img > 0] = out_img[out_img > 0]
    batch_images[track_index] = (road_img*255).astype(np.uint8)
  return batch_images

def compute_embeddings(batch_images, cnn_models):
  """
  Parameters:
    batch_images: nparray of shape(B, 224,448, 3)
    cnn_models: dictionary from model_names to models

  Returns:
     a dictionary from model_names to embeddings of shape(B, out_embedding_size)
  """
  # evaluate the pre-trained CNN embeddings
  model_embeddings = {}
  for model_name, model in cnn_models.items():
    model_embedding = model.predict(tf.convert_to_tensor(batch_images)) # Outputs (B,7,7,2048)
    model_embedding = model_embedding[:,1:-1,1:-1,:]
    model_embedding = tf.keras.layers.GlobalAveragePooling2D()(model_embedding) # Outputs (B,2048)
    model_embeddings[model_name] = model_embedding.numpy()
  return model_embeddings