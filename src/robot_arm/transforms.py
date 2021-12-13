import numpy as np

def build_rot_x(theta: float):
  return np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta), np.cos(theta)],
  ])

def build_rot_y(theta: float):
  return np.array([
    [np.cos(theta), 0, np.sin(theta)],
    [0, 1, 0],
    [-np.sin(theta), 0, np.cos(theta)],
  ])

def build_rot_z(theta: float):
  return np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1],
  ])

def down_arm_single_pt(points, rot, rot_z, tran):
  return down_arm(points[:, None], rot, rot_z, tran).reshape(-1)

def down_arm_multi_pt(points, rot, rot_z, tran):
  return down_arm(points.T, rot, rot_z, tran).T

def down_arm(points, rot, rot_z, tran):
  '''
  points = np.array([pt1, pt2, pt3, ...]).T
  '''
  return rot_z @ rot @ (points + tran[:, None])

def up_arm_single_pt(points, rot, rot_z, tran):
  return up_arm(points[:, None], rot, rot_z, tran).reshape(-1)

def up_arm_multi_pt(points, rot, rot_z, tran):
  return up_arm(points.T, rot, rot_z, tran).T

def up_arm(points, rot, rot_z, tran):
  return rot.T @ rot_z.T @ points - tran[:, None]
