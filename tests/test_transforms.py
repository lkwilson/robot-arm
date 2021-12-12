import numpy as np
from robot_arm.transforms import RobotState, ArmPoint, rot_x, rot_y, rot_z, down_arm, up_arm

'''
class RobotState:
  theta: np.ndarray
  arm_rots: list[np.ndarray]
  arm_trans: list[np.ndarray]
class ArmPoint:
  index: int
  point: np.ndarray
'''


def get_state():
  return RobotState(
    arm_rots = [
      rot_x(np.pi / 4),
      rot_y(np.pi / 4),
      rot_z(np.pi / 4),
      rot_z(np.pi / 4),
      rot_y(np.pi / 4),
      rot_x(np.pi / 4),
    ],
    arm_trans = [
      np.array([0, 0, 10]) for _ in range(6)
    ],
    thetas = np.zeros(6)
  )

def test_rot_x():
  assert False

def test_rot_y():
  assert False

def test_rot_z():
  assert False

def test_down_arm():
  ''' def down_arm(state: RobotState, arm_point: ArmPoint): '''
  assert False

def test_down_arm():
  ''' def up_arm(state: RobotState, arm_point: ArmPoint): '''
  assert False
