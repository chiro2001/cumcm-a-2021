import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# 主索节点的坐标和编号
data1 = pd.read_csv("data/附件1.csv", encoding='ANSI')
# print('主索节点的坐标和编号:\n', data1)
nodes_data = {}
for d in data1.itertuples():
    nodes_data[d[1]] = {
        # 'position': tuple(d[2:]),
        'position_raw': np.array(d[2:]),
        'position': np.array(d[2:]),
        # 伸缩量，即所有需要求解的变量
        'expand': 0
    }

# 促动器下端点（地锚点）坐标、
# 基准态时上端点（顶端）的坐标，
# 以及促动器对应的主索节点编号
data2 = pd.read_csv("data/附件2.csv", encoding='ANSI')
# print('data2:\n', data2)
for d in data2.itertuples():
    nodes_data[d[1]]['actuactor_head'] = np.array(d[2:5])
    nodes_data[d[1]]['actuactor_base'] = np.array(d[5:8])

# print(nodes_data)

triangles_data = []

# 反射面板对应的主索节点编号
data3 = pd.read_csv("data/附件3.csv", encoding='ANSI')
# print('data3:\n', data3)
for d in data3.itertuples():
    triangles_data.append(tuple(d[1:]))


# print(triangles_data)


# 绘制当前图像
def draw_points(points: np.ndarray = None, nodes_data_: dict = nodes_data):
    ax = plt.axes(projection='3d')
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    ax.set_zlim(-400, -100)

    if points is None:
        points = to_points(nodes_data_=nodes_data_)
    points2 = to_points(nodes_data_=nodes_data_, dict_key='actuactor_head')
    points3 = to_points(nodes_data_=nodes_data_, dict_key='actuactor_base')

    m = get_rotation_matrix(np.pi / 6, np.pi / 4, np.pi / 12)
    # m = get_rotation_matrix(0, 0, np.pi / 12)
    # m = get_rotation_matrix(a, b, c)

    # points = points * m
    # np.zeros((3, 3)) * np.zeros((100, 3)).T
    # 矩阵乘法用 np.dot...
    points = np.dot(points, m)

    ax.scatter3D(points.T[0], points.T[1], points.T[2], c="g", marker='.')
    ax.scatter3D(points2.T[0], points2.T[1], points2.T[2], c="c", marker='.')
    ax.scatter3D(points3.T[0], points3.T[1], points3.T[2], c='m', marker='.')
    plt.show()


# 计算在当前伸缩值状态下，主索节点的位置(position)
def update_expand(nodes_data_: dict = nodes_data):
    for name in nodes_data_:
        node = nodes_data_[name]
        # 用促动器下端坐标和最初的主索节点位置确定方向向量，计算伸缩量
        n = get_unit_vector(node['position_raw'], node['actuactor_base'])
        # 更新 position
        node['position'] = node['position_raw'] + n * node['expand']


# 转换数据到坐标
def to_points(nodes_data_: dict = nodes_data, dict_key: str = 'position') -> np.ndarray:
    points = []
    for name in nodes_data_:
        node = nodes_data_[name]
        points.append(node[dict_key])
    return np.array(points)


def do_rotation(alpha: float, beta: float) -> np.ndarray:
    m = get_rotation_matrix(0, alpha, beta)


for i in range(0, 8):
    # c -> beta
    # b -> alpha ?
    draw_points(b=0, c=0, a=i * np.pi / 8)
# import random
# for name in nodes_data:
#   nodes_data[name]['expand'] = 20 * random.random()
# update_expand()
# draw_points()
# print(plane_symmetry_point([1, 1, 1, 0], [2, 2, 2]))
