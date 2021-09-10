import numpy as np
import sympy
import copy


# 拿到单位方向向量(pt1 -> pt2)
def get_unit_vector(pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
    a = pt1 - pt2
    n = a / np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return n


# 三个点确定一个平面
def triangle_to_plane(points: np.ndarray) -> np.ndarray:
    # points = [nodes_data[code]['position'] for code in triangle]
    # Ax + By + Cz + D = 0
    # A = (points[2][1] - points[0][1]) * (points[2][2] - points[0][2]) - \
    #     (points[1][2] - points[0][2]) * (points[2][1] - points[0][1])
    # B = (points[2][0] - points[0][0]) * (points[1][2] - points[0][2]) - \
    #     (points[1][0] - points[0][0]) * (points[2][2] - points[0][2])
    # C = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - \
    #     (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    # D = -(A * points[0][0] + B * points[0][1] + C * points[0][2])
    A = (points[1][1] - points[0][1]) * (points[2][2] - points[0][2]) - \
        (points[1][2] - points[0][2]) * (points[2][1] - points[0][1])
    B = (points[2][0] - points[0][0]) * (points[1][2] - points[0][2]) - \
        (points[1][0] - points[0][0]) * (points[2][2] - points[0][2])
    C = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - \
        (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    D = -(A * points[0][0] + B * points[0][1] + C * points[0][2])
    return np.array((A, B, C, D))


# 求点到平面的对称点
def plane_symmetry_point(plane: np.ndarray, point: np.ndarray) -> np.ndarray:
    A, B, C, D = plane
    # x1, y1, z1 = point
    x2, y2, z2 = sympy.Symbol('x2'), sympy.Symbol('y2'), sympy.Symbol('z2')
    # ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
    # (x1 + x2) / 2 * A + (y1 + y2) / 2 * B + (z1 + z2) / 2 * C + D = 0
    # (x1 + x2) * A + (y1 + y2) * B + (z1 + z2) * C + 2 * D = 0
    # (p - point) == l * (A, B, C)
    l = sympy.Symbol('l')
    # print(sympy.Matrix(point).shape)
    # print(sympy.Matrix([x2, y2, z2]).shape)
    # print(sympy.Matrix([A, B, C]).shape)
    result = sympy.solve([sympy.Matrix(point) - sympy.Matrix([x2, y2, z2]) - l * sympy.Matrix([A, B, C]),
                          np.dot(np.array(point) + np.array([x2, y2, z2]), np.array([A, B, C])) + 2 * D],
                         [x2, y2, z2, l])
    # print(result, type(result))
    return np.array([result[x2], result[y2], result[z2]])


# 从方位角和仰角得到目标向量，单位度
def get_target_vector(alpha: float, beta: float, unit_degree: bool = True, r: float = 1) -> np.ndarray:
    if unit_degree:
        alpha = alpha / 360 * 2 * np.pi
        beta = beta / 360 * 2 * np.pi
    return np.array([
        r * np.sin(alpha) * np.cos(beta),
        r * np.sin(alpha) * np.sin(beta),
        r * np.cos(alpha)
    ])


# 得到三维旋转矩阵，单位弧度
# https://blog.csdn.net/fireflychh/article/details/82352710
def get_rotation_matrix(yaw: float, pitch: float, roll: float):
    from numpy import cos, sin
    return np.array([
        [cos(yaw) * cos(pitch),
         cos(yaw) * sin(pitch) * sin(roll) - sin(yaw) * cos(roll),
         cos(yaw) * sin(pitch) * cos(roll) + sin(yaw) * sin(roll)],
        [sin(yaw) * cos(pitch),
         sin(yaw) * sin(pitch) * sin(roll) + cos(yaw) * cos(roll),
         sin(yaw) * sin(pitch) * cos(roll) - cos(yaw) * sin(roll)],
        [-sin(pitch),
         cos(pitch) * sin(roll),
         cos(pitch) * cos(roll)]
    ])


def within(val, range_):
    r = copy.deepcopy(range_)
    list.sort(r)
    return r[0] <= val <= r[1]


def is_in_board(points: np.ndarray) -> bool:
    plane = triangle_to_plane(points)
    # (0, 0, z): C * z + D = 0, z = - D / C
    z = - plane[3] / plane[2]
    return sum([within(0, [np.min(points.T[0]), np.max(points.T[0])]),
                within(0, [np.min(points.T[1]), np.max(points.T[1])]),
                within(z, [np.min(points.T[2]), np.max(points.T[2])])]) == 3


def get_board_center(board: np.ndarray) -> np.ndarray:
    return (board[0] + board[1] + board[2]) / len(board)


def get_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.sqrt(np.sum((p2 - p1) ** 2))


def normalizing(n: np.ndarray) -> np.ndarray:
    return n / np.sqrt(np.sum(n ** 2))
