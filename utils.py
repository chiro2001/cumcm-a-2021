# 工具类
import numpy as np
import torch


# 拿到单位方向向量(pt1 -> pt2)
def get_unit_vector(pt1, pt2):
    a = pt1 - pt2
    n = a / torch.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    return n


# 三个点确定一个平面
def triangle_to_plane(points: torch.Tensor):
    n = torch.cross((points[1] - points[0]), (points[2] - points[0]))
    D = -torch.dot(n, points[0])
    return n, D


def get_rotation_matrix(alpha: float, beta: float):
    from numpy import cos, sin
    theta = np.pi / 2 - beta
    return torch.as_tensor(torch.mm(torch.as_tensor([
        [cos(theta), 0, -sin(theta)],
        [0, 1, 0],
        [sin(theta), 0, cos(theta)]
    ]), torch.as_tensor([
        [cos(alpha), sin(alpha), 0],
        [-sin(alpha), cos(alpha), 0],
        [0, 0, 1]
    ])), dtype=torch.float64)


def within(val, range_) -> bool:
    return range_[0] <= val <= range_[1]


def is_in_board(points: torch.Tensor) -> bool:
    n, D = triangle_to_plane(points)
    # (0, 0, z): C * z + D = 0, z = - D / C
    z = - D / n[2]
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
