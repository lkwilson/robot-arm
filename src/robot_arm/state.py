from dataclasses import dataclass
import numpy as np

from robot_arm import transforms

@dataclass
class RobotState:
  arm_rots: list[np.ndarray]
  ''' The arm rotations. frozen. index 0 is for the first arm, n=1. '''

  arm_trans: list[np.ndarray]
  ''' The arm translations. frozen. index 0 is for the first arm, n=1. '''

  thetas: list[float]
  ''' The each arm's current rotations. index 0 is for the first arm, n=1.  '''

  def __post_init__(self):
    self.arm_rots = [np.array(arm_rot) for arm_rot in self.arm_rots]
    self.arm_trans = [np.array(arm_tran) for arm_tran in self.arm_trans]
    self.thetas = np.array(self.thetas)

    assert len(self.arm_rots) == len(self.arm_trans)
    assert len(self.arm_rots) == len(self.thetas)

@dataclass
class ArmPoint:
  index: int
  ''' what arm's cartesian space this point exists in. 0 is robot's, 1 is arm 1. '''

  point: np.ndarray
  ''' the point '''

def down_arm(state: RobotState, arm_point: ArmPoint):
  ''' move arm_point down the arm '''
  # the current arm's n
  n = arm_point.index
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  new_point = transforms.build_rot_z(theta)@rot@(point + tran)
  return ArmPoint(n+1, new_point)

def up_arm(state: RobotState, arm_point: ArmPoint):
  ''' move arm_point up the arm. assumes arm_point.index > 0 '''
  # the previous arm's n
  n = arm_point.index - 1
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  # note: inverse of rotation/unitary matrix is just transpose
  new_point = rot.T @ transforms.build_rot_z(theta).T @ point - tran
  return ArmPoint(n, new_point)

def to_arm(state: RobotState, arm_point: ArmPoint, arm_number: int):
  ''' arm_number is 0 for robot 1 for first arm. return value will match arm_point.index '''
  point = arm_point
  while point.index < arm_number:
    point = down_arm(state, point)
  while point.index > arm_number:
    point = up_arm(state, point)
  return point
