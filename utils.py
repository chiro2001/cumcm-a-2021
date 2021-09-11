import numpy as np
import torch
import sympy
import copy


# 拿到单位方向向量(pt1 -> pt2)
def get_unit_vector(pt1, pt2):
    a = pt1 - pt2
    n = a / torch.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return n


# 三个点确定一个平面
def triangle_to_plane(points: torch.Tensor):
    # points = [nodes_data[code]['position'] for code in triangle]
    # Ax + By + Cz + D = 0
    # A = (points[2][1] - points[0][1]) * (points[2][2] - points[0][2]) - \
    #     (points[1][2] - points[0][2]) * (points[2][1] - points[0][1])
    # B = (points[2][0] - points[0][0]) * (points[1][2] - points[0][2]) - \
    #     (points[1][0] - points[0][0]) * (points[2][2] - points[0][2])
    # C = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - \
    #     (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    # D = -(A * points[0][0] + B * points[0][1] + C * points[0][2])

    # A = (points[1][1] - points[0][1]) * (points[2][2] - points[0][2]) - \
    #     (points[1][2] - points[0][2]) * (points[2][1] - points[0][1])
    # B = (points[2][0] - points[0][0]) * (points[1][2] - points[0][2]) - \
    #     (points[1][0] - points[0][0]) * (points[2][2] - points[0][2])
    # C = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - \
    #     (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    # D = -(A * points[0][0] + B * points[0][1] + C * points[0][2])
    # return torch.tensor((A, B, C, D))
    n = torch.cross((points[1] - points[0]), (points[2] - points[0]))
    D = -torch.dot(n, points[0])
    return n, D


# # 求点到平面的对称点
# def plane_symmetry_point(plane, point):
#     A, B, C, D = plane
#     # x1, y1, z1 = point
#     x2, y2, z2 = sympy.Symbol('x2'), sympy.Symbol('y2'), sympy.Symbol('z2')
#     # ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
#     # (x1 + x2) / 2 * A + (y1 + y2) / 2 * B + (z1 + z2) / 2 * C + D = 0
#     # (x1 + x2) * A + (y1 + y2) * B + (z1 + z2) * C + 2 * D = 0
#     # (p - point) == l * (A, B, C)
#     l = sympy.Symbol('l')
#     # print(sympy.Matrix(point).shape)
#     # print(sympy.Matrix([x2, y2, z2]).shape)
#     # print(sympy.Matrix([A, B, C]).shape)
#     result = sympy.solve([sympy.Matrix(point) - sympy.Matrix([x2, y2, z2]) - l * sympy.Matrix([A, B, C]),
#                           np.dot(np.array(point) + np.array([x2, y2, z2]), np.array([A, B, C])) + 2 * D],
#                          [x2, y2, z2, l])
#     # print(result, type(result))
#     return np.array([result[x2], result[y2], result[z2]])


# # 从方位角和仰角得到目标向量，单位度
# def get_target_vector(alpha: float, beta: float, unit_degree: bool = True, r: float = 1) -> np.ndarray:
#     if unit_degree:
#         alpha = alpha / 360 * 2 * np.pi
#         beta = beta / 360 * 2 * np.pi
#     return np.array([
#         r * np.sin(alpha) * np.cos(beta),
#         r * np.sin(alpha) * np.sin(beta),
#         r * np.cos(alpha)
#     ])


# 得到三维旋转矩阵，单位弧度
# https://blog.csdn.net/fireflychh/article/details/82352710
def get_rotation_matrix(yaw: float, pitch: float, roll: float):
    from torch import cos, sin
    return torch.tensor([
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


# def within(val, range_):
#     r = copy.deepcopy(range_)
#     list.sort(r)
#     return r[0] <= val <= r[1]
# def within(val, range_):
#     r = torch.tensor([range_[0] <= val and val <= range_[1], ])
#     return r
def within(val, range_) -> bool:
    # r = torch.tensor([range_[0] <= val and val <= range_[1], ])
    return range_[0] <= val <= range_[1]


def is_in_board(points: torch.Tensor) -> bool:
    n, D = triangle_to_plane(points)
    # (0, 0, z): C * z + D = 0, z = - D / C
    z = - D / n[2]
    # z = z.detach().numpy()
    # return sum([within(0, [torch.min(points.T[0]), torch.max(points.T[0])]),
    #             within(0, [torch.min(points.T[1]), torch.max(points.T[1])]),
    #             within(z, [torch.min(points.T[2]), torch.max(points.T[2])])]) == 3

    # return sum([within(0, [torch.min(points.transpose(0, 1)[0]), torch.max(points.transpose(0, 1)[0])]),
    #             within(0, [torch.min(points.transpose(0, 1)[1]), torch.max(points.transpose(0, 1)[1])]),
    #             within(z, [torch.min(points.transpose(0, 1)[2]), torch.max(points.transpose(0, 1)[2])])]) == 3
    # p = points.detach().numpy()
    p = points
    return sum([within(0, [min([p[0][0], p[1][0], p[2][0]]),
                           max([p[0][0], p[1][0], p[2][0]])]),
                within(0, [min([p[0][1], p[1][0], p[2][1]]),
                           max([p[0][1], p[1][0], p[2][1]])]),
                within(z, [min([p[0][2], p[1][0], p[2][2]]),
                           max([p[0][2], p[1][0], p[2][2]])])]) == 3


def get_board_center(board):
    return (board[0] + board[1] + board[2]) / len(board)


def get_distance(p1, p2) -> float:
    return torch.sqrt(torch.sum((p2 - p1) ** 2))


def normalizing(n):
    return n / torch.sqrt(torch.sum(n ** 2))
