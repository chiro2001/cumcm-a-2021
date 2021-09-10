import pandas as pd
import torch
from torch import nn
from utils import *


class FAST(nn.Module):
    def __init__(self):
        super(FAST, self).__init__()

        self.count_nodes: int = None
        self.count_triangles: int = None
        self.expands: nn.Parameter = None
        self.position_raw: np.ndarray = None
        self.position: np.ndarray = None
        self.actuator_head: np.ndarray = None
        self.actuator_base: np.ndarray = None
        self.triangles_data: np.ndarray = None
        self.name_list: list = []
        self.index: dict = {}
        self.paddings_raw: np.ndarray = None

        self.loss_weights: np.ndarray = np.array([1, 1, 1])

        self.read_data()

        self.paddings_raw = self.get_paddings(source=self.position_raw)

    def get_edge(self, node1: str, node2: str, source: np.ndarray) -> float:
        return get_distance(source[self.index[node1]], source[self.index[node2]])

    # 得到点与点的间隔
    # 因为不知道具体边的关系，就按照三角形三条边考虑吧，数据是原来三倍
    def get_paddings(self, source: np.ndarray = None) -> np.ndarray:
        source = source if source is not None else self.position_raw
        paddings = []
        for triangle in self.triangles_data:
            edges = [self.get_edge(triangle[0], triangle[1], source),
                     self.get_edge(triangle[1], triangle[2], source),
                     self.get_edge(triangle[2], triangle[0], source)]
            paddings.append(edges)
        return paddings

    def read_data(self):
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
            nodes_data[d[1]]['actuator_head'] = np.array(d[2:5])
            nodes_data[d[1]]['actuator_base'] = np.array(d[5:8])

        # print(nodes_data)

        triangles_data = []
        # 反射面板对应的主索节点编号
        data3 = pd.read_csv("data/附件3.csv", encoding='ANSI')
        # print('data3:\n', data3)
        for d in data3.itertuples():
            triangles_data.append(tuple(d[1:]))
        # print(triangles_data)

        # 主索节点名称到下标号的映射
        self.name_list = list(nodes_data.keys())
        self.index = {self.name_list[i]: i for i in range(len(self.name_list))}
        self.position_raw = [nodes_data[name]['position_raw'] for name in self.name_list]
        self.position = np.array([nodes_data[name]['position'] for name in self.name_list])
        self.actuator_head = np.array([nodes_data[name]['actuator_head'] for name in self.name_list])
        self.actuator_base = np.array([nodes_data[name]['actuator_base'] for name in self.name_list])
        self.triangles_data = triangles_data
        self.count_triangles = len(triangles_data)
        self.count_nodes = len(list(nodes_data.keys()))
        # 每个节点的伸展长度
        self.expands = nn.Parameter(torch.tensor(np.zeros(len(list(nodes_data.keys())))))

    # 计算在当前伸缩值状态下，主索节点的位置
    def update_position(self):
        for i in range(self.count_nodes):
            n = get_unit_vector(self.position_raw[i], self.actuator_base[i])
            self.position[i] = self.position_raw[i] + n * self.expands[i]

    # 判断当前伸缩条件下能否满足伸缩量限制
    def is_expands_legal(self) -> bool:
        return torch.max(self.expands) > 0.6 or torch.min(self.expands) < -0.6

    # 判断当前伸缩条件能否满足间隔限制
    def is_padding_legal(self, paddings: np.ndarray = None) -> bool:
        paddings = self.get_paddings(source=self.position) if paddings is None else paddings
        return np.sum([
            np.sum([
                (1 if (np.abs(paddings[i][j] - self.paddings_raw[i][j]) < 0.0007 * self.paddings_raw[i][j]) else 0) for
                j in range(3)
            ]) for i in range(len(paddings))
        ]) == 0

    # 得到间隔误差
    def get_padding_loss(self, weight: float = 1) -> float:
        paddings = self.get_paddings(source=self.position)
        if self.is_padding_legal(paddings=paddings):
            return 0
        return np.sum([
            np.array([
                np.sum([
                    (paddings[i][j] - self.paddings_raw[i][j]) ** 2 for j in range(3)
                ]) for i in range(len(paddings))
            ]) ** 2
        ]) * weight

    # 得到伸缩误差
    def get_expand_loss(self, weight: float = 1) -> float:
        return np.sum([0 if (-0.6 <= expand <= 0.6) else (np.abs(expand) - 0.6) for expand in self.expands]) * weight

    # 得到光通量误差
    def get_light_loss(self, weight: float = 1) -> float:
        return 0 * weight

    # 得到拟合精度误差
    def get_fitting_loss(self, weight: float = 1) -> float:
        for triangle in self.triangles_data:
            board = self.get_board(triangle)
        return 0 * weight

    # 计算整体误差
    def get_loss(self) -> float:
        return self.get_expand_loss(weight=self.loss_weights[0]) + self.get_padding_loss(
            weight=self.loss_weights[0]) + self.get_fitting_loss(weight=self.loss_weights[1]) + self.get_light_loss(
            weight=self.loss_weights[2])

    def get_board(self, triangle) -> np.ndarray:
        return np.array([self.position[self.index[triangle[i]]] for i in range(3)])

    def get_boards(self) -> list:
        return [self.get_board(triangle) for triangle in self.triangles_data]

    # 取得变换后经过 z 轴的板子作为顶点参考板
    def get_center_board(self) -> np.ndarray:
        boards = self.get_boards()
        for board in boards:
            if is_in_board(board):
                return board
        return None

    # 得到抛物面的顶点
    def get_vertex(self) -> float:
        board = self.get_center_board()
        if board is None:
            raise Exception("取得顶点参考板错误！")
        plane = triangle_to_plane(board)
        return -plane[3] / plane[2]

    # 计算顶点位置
    def forward(self):
        self.update_position()
        return self.position


def main():
    model = FAST()


if __name__ == '__main__':
    main()
