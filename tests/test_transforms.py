import numpy as np
import logging
from robot_arm.transforms import build_rot_x, build_rot_y, build_rot_z
from robot_arm.state import RobotState, ArmPoint, down_arm, up_arm, to_arm
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
      build_rot_x(np.pi / 2),
      build_rot_y(np.pi / 2),
      build_rot_z(np.pi / 2),
      build_rot_z(np.pi / 2),
      build_rot_y(np.pi / 2),
      build_rot_x(np.pi / 2),
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
    assert np.all(np.isclose(rot.as_matrix(), build_rot_x(theta)))

def test_rot_y():
  x_hat = np.array([0, 1, 0])
  for theta in np.arange(start=-20, stop=20, step=.05):
    rot = Rotation.from_rotvec(theta * x_hat)
    logger.info("%s", rot.as_matrix())
    assert np.all(np.isclose(rot.as_matrix(), build_rot_y(theta)))

def test_rot_z():
  x_hat = np.array([0, 0, 1])
  for theta in np.arange(start=-20, stop=20, step=.05):
    rot = Rotation.from_rotvec(theta * x_hat)
    logger.info("%s", rot.as_matrix())
    assert np.all(np.isclose(rot.as_matrix(), build_rot_z(theta)))

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

def test_high_traverse_arm():
  state = get_state()
  state.thetas = np.array([1, 2, 3, 4, 5, 6])
  init_point = ArmPoint(0, np.zeros(3))
  point = init_point
  for i in range(6):
    point = down_arm(state, point)
  quick_point = to_arm(state, point, 6)
  logger.info('%s from %s', quick_point, point)
  assert np.all(np.isclose(quick_point.point, point.point))
  quick_point = to_arm(state, init_point, 6)
  assert np.all(np.isclose(quick_point.point, point.point))

  back_point = to_arm(state, point, 2)
  back_point = to_arm(state, point, 5)
  back_point = to_arm(state, point, 6)
  back_point = to_arm(state, point, 0)
  assert np.all(np.isclose(back_point.point, init_point.point))
