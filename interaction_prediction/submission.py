# InteractionSubmission:
import tensorflow as tf
import numpy as np
from waymo_open_dataset.protos import motion_submission_pb2

def eval_and_generate_submission(model, eval_dataset, num_modes = 6):
  submission = motion_submission_pb2.MotionChallengeSubmission()
  submission.submission_type = 2
  submission.affiliation = ''
  submission.description = ''
  submission.method_link = ''

  current_scenario_id = ""
  for i, batch in enumerate(eval_dataset):
    if i % 100 == 0:
      print(i)
    trajectories, confidences = model.predict_step(batch) # (B, K**2, 2, 16, 2), (B, K**2)
    trajectories = trajectories.numpy()
    confidences = confidences.numpy()
    for example_index in range(len(trajectories)):
      example_scenario_id = batch['scenario_id'][example_index, 0].numpy()
      if example_scenario_id != current_scenario_id:
        current_scenario_id = example_scenario_id
        current_scenario_prediction = submission.scenario_predictions.add()
        current_scenario_prediction.scenario_id = example_scenario_id
       # prediction is of type SingleObjectPrediction
      for mode in range(num_modes):
        joint_trajectory = current_scenario_prediction.joint_prediction.joint_trajectories.add()
        joint_trajectory.confidence = confidences[example_index, mode]
        for j in batch['indices'][example_index]:
          object_trajectory = joint_trajectory.trajectories.add()
          object_trajectory.object_id = batch['object_id'][example_index, j].numpy().astype(np.int32)
          center_x = []
          center_y = []
          for t in range(16):
              x = trajectories[example_index, mode, j, t, 0]
              y = trajectories[example_index, mode, j, t, 1]
              center_x.append(x)
              center_y.append(y)
          trajectory = object_trajectory.trajectory
          trajectory.center_x.extend(center_x)
          trajectory.center_y.extend(center_y)
  return submission

def write_submission_to_file(submission, output_filename):
  f = open(output_filename, "wb")
  f.write(submission.SerializeToString())
  f.close()