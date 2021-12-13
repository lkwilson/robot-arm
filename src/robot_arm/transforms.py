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

def down_arm(point, rot, rot_z, tran):
  return rot_z @ rot @ (point + tran)

def up_arm(point, rot, rot_z, tran):
  return rot.T @ rot_z.T @ point - tran
