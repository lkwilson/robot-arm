import numpy as np
import logging
from robot_arm.transforms import RobotState, ArmPoint, rot_x, rot_y, rot_z, down_arm, up_arm
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

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
      rot_x(np.pi / 2),
      rot_y(np.pi / 2),
      rot_z(np.pi / 2),
      rot_z(np.pi / 2),
      rot_y(np.pi / 2),
      rot_x(np.pi / 2),
    ],
    arm_trans = [
      np.array([0, 0, 10]) for _ in range(6)
    ],
    thetas = np.zeros(6)
  )

def rot_check(rot, x, exp):
  res = rot @ x
  logger.info(res)
  assert np.all(np.isclose(res, exp))

def test_rot_x():
  x_hat = np.array([1, 0, 0])
  for theta in np.arange(start=-20, stop=20, step=.05):
    rot = Rotation.from_rotvec(theta * x_hat)
    logger.info("%s", rot.as_matrix())
    assert np.all(np.isclose(rot.as_matrix(), rot_x(theta)))

def test_rot_y():
  x_hat = np.array([0, 1, 0])
  for theta in np.arange(start=-20, stop=20, step=.05):
    rot = Rotation.from_rotvec(theta * x_hat)
    logger.info("%s", rot.as_matrix())
    assert np.all(np.isclose(rot.as_matrix(), rot_y(theta)))

def test_rot_z():
  x_hat = np.array([0, 0, 1])
  for theta in np.arange(start=-20, stop=20, step=.05):
    rot = Rotation.from_rotvec(theta * x_hat)
    logger.info("%s", rot.as_matrix())
    assert np.all(np.isclose(rot.as_matrix(), rot_z(theta)))

def test_traverse_arm():
  ''' def down_arm(state: RobotState, arm_point: ArmPoint): '''
  state = get_state()
  init_point = ArmPoint(0, np.zeros(3))
  point = init_point
  for i in range(6):
    last_point = point
    point = down_arm(state, point)
    assert point.index == last_point.index + 1
    logger.info('%s -> %s', last_point, point)
  end_point = point
  assert point.index == 6
  for i in range(6):
    last_point = point
    point = up_arm(state, point)
    assert point.index == last_point.index - 1
    logger.info('%s -> %s', last_point, point)
  assert init_point.index == point.index
  assert np.all(np.isclose(point.point, init_point.point))

  state.thetas = np.array([1, 2, 3, 4, 5, 6])
  for i in range(6):
    point = down_arm(state, point)
  assert not np.all(np.isclose(end_point.point, point.point))
  for i in range(6):
    point = up_arm(state, point)
  assert init_point.index == point.index
  assert np.all(np.isclose(point.point, init_point.point))
