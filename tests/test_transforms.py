import numpy as np
import logging
from dataclasses import dataclass
from robot_arm import transforms
from robot_arm.transforms import build_rot_x, build_rot_y, build_rot_z
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class TestRobotState:
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
class TestArmPoint:
  index: int
  ''' what arm's cartesian space this point exists in. 0 is robot's, 1 is arm 1. '''

  point: np.ndarray
  ''' the point '''

def down_arm(state: TestRobotState, arm_point: TestArmPoint):
  ''' move arm_point down the arm '''
  # the current arm's n
  n = arm_point.index
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  new_point = transforms.build_rot_z(theta)@rot@(point + tran)
  return TestArmPoint(n+1, new_point)

def up_arm(state: TestRobotState, arm_point: TestArmPoint):
  ''' move arm_point up the arm. assumes arm_point.index > 0 '''
  # the previous arm's n
  n = arm_point.index - 1
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  # note: inverse of rotation/unitary matrix is just transpose
  new_point = rot.T @ transforms.build_rot_z(theta).T @ point - tran
  return TestArmPoint(n, new_point)

def to_arm(state: TestRobotState, arm_point: TestArmPoint, arm_number: int):
  ''' arm_number is 0 for robot 1 for first arm. return value will match arm_point.index '''
  point = arm_point
  while point.index < arm_number:
    point = down_arm(state, point)
  while point.index > arm_number:
    point = up_arm(state, point)
  return point

def down_arm(state: TestRobotState, arm_point: TestArmPoint):
  ''' move arm_point down the arm '''
  # the current arm's n
  n = arm_point.index
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  new_point = transforms.build_rot_z(theta)@rot@(point + tran)
  return TestArmPoint(n+1, new_point)

def up_arm(state: TestRobotState, arm_point: TestArmPoint):
  ''' move arm_point up the arm. assumes arm_point.index > 0 '''
  # the previous arm's n
  n = arm_point.index - 1
  theta = state.thetas[n]
  rot = state.arm_rots[n]
  tran = state.arm_trans[n]
  point = arm_point.point

  # note: inverse of rotation/unitary matrix is just transpose
  new_point = rot.T @ transforms.build_rot_z(theta).T @ point - tran
  return TestArmPoint(n, new_point)

def to_arm(state: TestRobotState, arm_point: TestArmPoint, arm_number: int):
  ''' arm_number is 0 for robot 1 for first arm. return value will match arm_point.index '''
  point = arm_point
  while point.index < arm_number:
    point = down_arm(state, point)
  while point.index > arm_number:
    point = up_arm(state, point)
  return point


def get_state():
  return TestRobotState(
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
  ''' def down_arm(state: TestRobotState, arm_point: TestArmPoint): '''
  state = get_state()
  init_point = TestArmPoint(0, np.zeros(3))
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
  init_point = TestArmPoint(0, np.zeros(3))
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

def test_multi_dim_trans():
  pts = np.array([
    [1, 2, 3],
    [3, 4, 5],
    [1, 2, 8],
    [1, 2, 8],
  ])
  down_target_pts = np.array([
    [1, 2, 13],
    [3, 4, 15],
    [1, 2, 18],
    [1, 2, 18],
  ])
  logger.info("pre: %s", pts)
  down_res = transforms.down_arm_multi_pt(pts, transforms.build_rot_x(0), transforms.build_rot_z(0), np.array([0, 0, 10]))
  logger.info("down: %s", down_res)
  assert np.all(np.isclose(down_res, down_target_pts))
  up_res = transforms.up_arm_multi_pt(down_res, transforms.build_rot_x(0), transforms.build_rot_z(0), np.array([0, 0, 10]))
  logger.info("up: %s", up_res)
  assert np.all(np.isclose(up_res, pts))

def test_single_dim_trans():
  pts = np.array([1, 2, 3])
  logger.info("pre: %s", pts)
  down_res = transforms.down_arm_single_pt(pts, transforms.build_rot_x(0), transforms.build_rot_z(0), np.array([0, 0, 10]))
  logger.info("down: %s", down_res)
  assert np.all(np.isclose(down_res, [1, 2, 13]))
  up_res = transforms.up_arm_single_pt(down_res, transforms.build_rot_x(0), transforms.build_rot_z(0), np.array([0, 0, 10]))
  logger.info("up: %s", up_res)
  assert np.all(np.isclose(up_res, pts))
