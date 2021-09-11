import sys
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import torch.optim as optim
from utils import *


class FAST(nn.Module):
    # 一些常量设定
    R = 300
    F = 0.466
    R_SURFACE = 150

    def __init__(self, device: torch.device):
        super(FAST, self).__init__()

        self.device: torch.device = device
        self.count_nodes: int = None
        self.count_triangles: int = None
        self.expands: nn.Parameter = None
        self.position_raw = None
        self.position: torch.Tensor = None
        self.actuator_head: torch.Tensor = None
        self.actuator_base: torch.Tensor = None
        self.triangles_data: list = None
        self.name_list: list = []
        self.index: dict = {}
        self.paddings_raw: list = None
        self.unit_vectors: torch.Tensor = None
        self.boards: torch.Tensor = None

        self.loss_weights = torch.tensor([1, 1, 1]).to(self.device)

        self.read_data()

        self.paddings_raw = self.get_paddings(source=self.position_raw)

        self.to(self.device)
        # print(self.expands, self.unit_vectors)

    def get_edge(self, node1: str, node2: str, source) -> float:
        return get_distance(source[self.index[node1]], source[self.index[node2]])

    # 得到点与点的间隔
    # 因为不知道具体边的关系，就按照三角形三条边考虑吧，数据（大约）是原来三倍
    def get_paddings(self, source: torch.Tensor = None) -> torch.Tensor:
        source = source if source is not None else self.position_raw
        paddings = []
        for triangle in self.triangles_data:
            edges = torch.stack([self.get_edge(triangle[0], triangle[1], source),
                                 self.get_edge(triangle[1], triangle[2], source),
                                 self.get_edge(triangle[2], triangle[0], source)])
            paddings.append(edges)
        return torch.stack(paddings)

    def read_data(self):
        # 主索节点的坐标和编号
        data1 = pd.read_csv("data/附件1.csv", encoding='ANSI')
        # print('主索节点的坐标和编号:\n', data1)
        nodes_data = {}
        for d in data1.itertuples():
            nodes_data[d[1]] = {
                # 'position': tuple(d[2:]),
                'position_raw': d[2:],
                'position': d[2:],
                # 伸缩量，即所有需要求解的变量
                'expand': 0
            }

        # 促动器下端点（地锚点）坐标、
        # 基准态时上端点（顶端）的坐标，
        # 以及促动器对应的主索节点编号
        data2 = pd.read_csv("data/附件2.csv", encoding='ANSI')
        # print('data2:\n', data2)
        for d in data2.itertuples():
            nodes_data[d[1]]['actuator_head'] = d[2:5]
            nodes_data[d[1]]['actuator_base'] = d[5:8]

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
        self.position_raw = torch.from_numpy(
            np.array([nodes_data[name]['position_raw'] for name in self.name_list])).to(self.device)
        # print([nodes_data[name]['position'] for name in self.name_list])
        self.position = torch.from_numpy(np.array([nodes_data[name]['position'] for name in self.name_list])).to(
            self.device)
        self.actuator_head = torch.from_numpy(
            np.array([nodes_data[name]['actuator_head'] for name in self.name_list])).to(self.device)
        self.actuator_base = torch.from_numpy(
            np.array([nodes_data[name]['actuator_base'] for name in self.name_list])).to(self.device)
        self.triangles_data = triangles_data
        self.count_triangles = len(triangles_data)
        self.count_nodes = len(list(nodes_data.keys()))
        # self.unit_vectors = torch.from_numpy(np.array(
        #     [get_unit_vector(self.position_raw[i], self.actuator_base[i]).cpu().numpy() for i in
        #      range(self.count_nodes)]).T).to(self.device)
        self.unit_vectors = torch.from_numpy(np.array(
            [get_unit_vector(self.position_raw[i], self.actuator_base[i]).cpu().numpy() for i in
             range(self.count_nodes)])).to(self.device)
        # 每个节点的伸展长度
        # self.expands = nn.Parameter(torch.zeros(self.count_nodes, dtype=torch.float64))
        self.expands = nn.Parameter(torch.rand(self.count_nodes, dtype=torch.float64) * 1.2 - 0.6)
        # 准备好 boards 数据
        self.boards = self.get_boards().to(self.device)

    # 计算在当前伸缩值状态下，主索节点的位置
    def update_position(self):
        # print(self.position_raw, self.unit_vectors, self.expands)
        # print(self.unit_vectors, self.expands)
        # print(self.unit_vectors.shape, self.expands.shape)
        # self.position = self.position_raw + torch.dot(self.unit_vectors,
        #                                               torch.reshape(self.expands, (len(self.expands), 1)))
        m = self.unit_vectors * self.expands.reshape((self.count_nodes, 1))
        # m = torch.matmul(self.unit_vectors, self.expands)
        self.position = self.position_raw + m
        # print(self.position.shape)
        # for i in range(self.count_nodes):
        #     self.position[i] = self.position_raw[i] + self.unit_vectors[i] * self.expands[i]

    # 判断当前伸缩条件下能否满足伸缩量限制
    def is_expands_legal(self) -> bool:
        r = torch.max(self.expands) > 0.6 or torch.min(self.expands) < -0.6
        return r

    # 判断当前伸缩条件能否满足间隔限制
    def is_padding_legal(self, paddings=None) -> bool:
        paddings = self.get_paddings(source=self.position) if paddings is None else paddings
        return torch.sum(torch.tensor([
            torch.sum(torch.tensor([
                (1 if (torch.abs(paddings[i][j] - self.paddings_raw[i][j]) < 0.0007 * self.paddings_raw[i][j]) else 0)
                for j in range(3)
            ])) for i in range(len(paddings))
        ])) == 0

    # 得到间隔误差
    def get_padding_loss(self, weight: float = 1) -> float:
        paddings = self.get_paddings(source=self.position)
        if self.is_padding_legal(paddings=paddings):
            return 0
        # length = len(paddings)
        # r = torch.sum(torch.from_numpy(
        #     np.array([
        #         torch.sum(torch.tensor([
        #             (paddings[i][j] - self.paddings_raw[i][j]) ** 2 for j in range(3)
        #         ])).numpy() for i in range(length)
        #     ]) ** 2
        # ))
        r = torch.sum(((paddings - self.paddings_raw) ** 2).flatten())
        return r * weight

    # 得到伸缩误差
    def get_expand_loss(self, weight: float = 1) -> float:
        r = torch.sum(torch.stack(
            [torch.tensor(0).to(self.device) if (-0.6 <= expand <= 0.6) else (torch.abs(expand) - 0.6) for expand in
             self.expands]))
        return r * weight

    # 得到光通量误差
    def get_light_loss(self, weight: float = 1) -> float:
        return 0 * weight

    # 得到拟合精度误差
    def get_fitting_loss(self, weight: float = 1) -> float:
        loss_sum = 0
        count_surface = 0
        self.boards = self.get_boards()
        # 得到底部顶点
        vertex = self.get_vertex()
        for board in self.boards:
            # 反射板中心
            center = get_board_center(board)
            # 在半径 150m 以外的板子不计算拟合精度误差
            if center[0] ** 2 + center[1] ** 2 <= FAST.R_SURFACE ** 2:
                count_surface += 1
                n_plane, D = triangle_to_plane(board)
                # 反射板法向量
                n_board = n_plane
                # 抛物面方程
                # z = x**2 / (4 * f) + y**2 / (4 * f) + h,
                # zdx = x / (2 * f)
                # zdy = y / (2 * f)
                # zdz = -1
                # h = vertex,
                h = vertex
                # f = |h| - (1 - F) * R
                f = torch.abs(h) - (1 - FAST.F) * FAST.R
                # 解方程求反射板法线和抛物面交点
                # (x-x0) / A == (y-y0) / B == (z-z0) / C
                x0, y0, z0 = center
                A, B, C = n_board
                if A == 0.0:
                    A = A + 1e-6
                a = B ** 2 / (4 * f * A ** 2) + 1 / (4 * f)
                b = B * y0 / (2 * A * f) - C / A - B ** 2 * x0 / (2 * f * A ** 2)
                c = B ** 2 * x0 ** 2 / (4 * f * A ** 2) - B * y0 * x0 / (2 * A * f) + y0 ** 2 / (
                        4 * f) + h + C / A * x0 - z0
                result = [None, None]
                x = (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                result[0] = torch.stack([
                    x,
                    B / A * (x - x0) + y0,
                    C / A * (x - x0) + z0
                ])
                x = (-b - torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                result[1] = torch.stack([
                    x,
                    B / A * (x - x0) + y0,
                    C / A * (x - x0) + z0
                ])

                # print("result", result)
                # 选择近的那一个作为交点
                result_distance = [get_distance(center, r) for r in result]
                result_point = result[1] if result_distance[0] > result_distance[1] else result[0]
                # 抛物面法向量
                n_surface = torch.stack([
                    result_point[0] / (2 * f),
                    result_point[1] / (2 * f),
                    torch.tensor(-1).to(self.device)
                ])
                # 标准化向量
                n_surface, n_board = torch.abs(normalizing(n_surface)), torch.abs(normalizing(n_board))
                # 求点积
                dot = torch.dot(n_surface, n_board)
                loss_sum += dot
            else:
                continue

        r = (1 - (loss_sum / count_surface)) * weight
        return r

    # 计算整体误差
    def get_loss(self) -> float:
        return self.get_expand_loss(weight=self.loss_weights[0]) + \
               self.get_padding_loss(weight=self.loss_weights[0]) + \
               self.get_fitting_loss(weight=self.loss_weights[1]) + \
               self.get_light_loss(weight=self.loss_weights[2])

    def get_board(self, triangle) -> torch.Tensor:
        # return torch.from_numpy(np.array([self.position[self.index[triangle[i]]].detach().numpy() for i in range(3)]))
        r = torch.stack([self.position[self.index[triangle[i]]] for i in range(3)])
        return r

    def get_boards(self) -> torch.Tensor:
        r = torch.stack([self.get_board(triangle) for triangle in self.triangles_data])
        return r

    # 取得变换后经过 z 轴的板子作为顶点参考板
    def get_bottom_board(self):
        # boards = self.get_boards()
        boards = self.boards
        for board in boards:
            if is_in_board(board):
                return board
        return None

    # 得到抛物面的顶点
    def get_vertex(self) -> float:
        board = self.get_bottom_board()
        if board is None:
            raise Exception("取得顶点参考板错误！")
        n, D = triangle_to_plane(board)
        return -D / n[2]

    # 计算主索节点位置
    def forward(self):
        self.update_position()
        loss = self.get_loss()
        return loss


g_fig = None


# 绘制当前图像
def draw(model: FAST, wait_time: int = 0):
    global g_fig
    if wait_time < 0:
        if g_fig is not None:
            try:
                plt.close(g_fig)
            except Exception as e:
                print(e)

    ax = plt.axes(projection='3d')
    ax.view_init(elev=10., azim=11)
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    ax.set_zlim(-400, -100)

    points = model.position.detach().numpy()
    ax.scatter3D(points.T[0], points.T[1], points.T[2], c="g", marker='.')

    # if wait_time == 0:
    #     plt.show()
    if 0 <= wait_time:
        if g_fig is None:
            g_fig = plt.figure(1)
        plt.draw()
        if wait_time > 0:
            plt.pause(wait_time)
        else:
            # plt.show(block=False)
            plt.pause(wait_time + 0.01)
        # plt.close(g_fig)
    elif wait_time < 0:
        plt.draw()
    plt.clf()


def main(alpha: float, beta: float, learning_rate: float = 1e-4, plot_picture: bool = True, device: str = None,
         wait_time: int = 0):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FAST(device=device)
    # TODO: 旋转模型
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(1000):
        optimizer.zero_grad()
        loss = model()
        print(f'loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        print(model.expands)
        if plot_picture:
            draw(model, wait_time=wait_time)


if __name__ == '__main__':
    device_ = sys.argv[1] if len(sys.argv) >= 2 else None
    learning_rate_ = float(sys.argv[2]) if len(sys.argv) >= 3 else 1e-2
    main(0, 0, learning_rate=learning_rate_, plot_picture=True, device=device_, wait_time=0)
