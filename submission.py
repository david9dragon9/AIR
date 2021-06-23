import tensorflow as tf
import numpy as np
from waymo_open_dataset.protos import motion_submission_pb2

def eval_and_generate_submission(model, eval_dataset, num_modes = 6):
  submission = motion_submission_pb2.MotionChallengeSubmission()
  submission.submission_type = 1
  current_scenario_id = ""
  for i, batch in enumerate(eval_dataset):
    if i % 100 == 0:
      print(i)
    trajectories, confidences = model.predict_step(batch) # (B, 3, 16, 2), (B, 3)
    trajectories = trajectories.numpy()
    confidences = confidences.numpy()
    for example_index in range(len(trajectories)):
      example_scenario_id = batch['scenario_id'][example_index][0].numpy()
      if example_scenario_id != current_scenario_id:
        current_scenario_id = example_scenario_id
        current_scenario_prediction = submission.scenario_predictions.add()
        current_scenario_prediction.scenario_id = example_scenario_id
      prediction = current_scenario_prediction.single_predictions.predictions.add() # prediction is of type SingleObjectPrediction
      prediction.object_id = batch['object_id'][example_index][0].numpy().astype(np.int32)
      for mode in range(num_modes):
        scored_trajectory = prediction.trajectories.add()
        scored_trajectory.confidence = confidences[example_index, mode]
        center_x = []
        center_y = []
        for t in range(16):
            x = trajectories[example_index, mode, t, 0]
            y = trajectories[example_index, mode, t, 1]
            center_x.append(x)
            center_y.append(y)
        trajectory = scored_trajectory.trajectory
        trajectory.center_x.extend(center_x)
        trajectory.center_y.extend(center_y)
  return submission

def write_submission_to_file(submission, output_filename):
  f = open(output_filename, "wb")
  f.write(submission.SerializeToString())
  f.close()

def combine_submission_bins(submission_files, output_bin):
  scenario_id_to_sps = {}
  for submission_file in submission_files:
    submission = motion_submission_pb2.MotionChallengeSubmission()
    with open("drive/MyDrive/Motion/submissions/" + submission_file, "rb") as f:
      submission.ParseFromString(f.read())
    for sp in submission.scenario_predictions:
      if sp.scenario_id in scenario_id_to_sps:
        scenario_id_to_sps[sp.scenario_id].append(sp)
      else:
        scenario_id_to_sps[sp.scenario_id] = [sp]
  submission = motion_submission_pb2.MotionChallengeSubmission()
  submission.submission_type = 1
  for scenario_id, sps in scenario_id_to_sps.items():
    scenario_prediction = submission.scenario_predictions.add()
    scenario_prediction.scenario_id = scenario_id
    for sp in sps:
      scenario_prediction.single_predictions.predictions.extend(sp.single_predictions.predictions)
  with open("drive/MyDrive/Motion/submissions/" + output_bin, "wb") as f:
    f.write(submission.SerializeToString())