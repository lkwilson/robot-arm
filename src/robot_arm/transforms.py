import numpy as np


'''  rot mat builders '''

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


''' helper functions for demo purposes really '''

def down_arm_single_pt(point, rot, rot_z, tran):
  ''' point has shape (3, ) '''
  return down_arm(point[:, None], rot, rot_z, tran).reshape(-1)

def down_arm_multi_pt(points, rot, rot_z, tran):
  ''' points has shape (n, 3) '''
  return down_arm(points.T, rot, rot_z, tran).T

def up_arm_single_pt(point, rot, rot_z, tran):
  ''' points has shape (3, ) '''
  return up_arm(point[:, None], rot, rot_z, tran).reshape(-1)

def up_arm_multi_pt(points, rot, rot_z, tran):
  ''' points has shape (n, 3) '''
  return up_arm(points.T, rot, rot_z, tran).T


''' the raw fast api '''

def up_arm(points, rot, rot_z, tran):
  '''
  points has shape (3, n)

  points = np.array([pt1, pt2, pt3, ...]).T
  '''
  return rot.T @ rot_z.T @ points - tran[:, None]

def down_arm(points, rot, rot_z, tran):
  '''
  points has shape (3, n)

  points = np.array([pt1, pt2, pt3, ...]).T
  '''
  return rot_z @ rot @ (points + tran[:, None])
