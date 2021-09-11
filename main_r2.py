import argparse
import pandas as pd
import time
import matplotlib.pyplot as plt
from torch import nn
import traceback
import torch.optim as optim
from tqdm import trange
import threading
from utils import *
import cv2
from base_logger import logger

draw_threaded: bool = False


class FAST(nn.Module):
    # 一些常量设定
    R = 300 + 0.4
    F = 0.466
    R_SURFACE = 150
    R_CABIN = 1.

    def __init__(self, **kwargs):
        super(FAST, self).__init__()

        device = torch.device(kwargs.get('device', None) if kwargs.get('device', None) is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"))
        self.init_randomly = kwargs.get('randomly_init', False)
        self.device: torch.device = device
        self.kwargs = kwargs

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
        self.paddings_raw: torch.Tensor = None
        self.unit_vectors: torch.Tensor = None
        self.boards: torch.Tensor = None

        self.loss_weights: torch.Tensor = torch.tensor([5, 2e3, 1e-4]).to(self.device)
        self.vertex: nn.Parameter = nn.Parameter(torch.tensor(-FAST.R)).to(self.device)

        self.read_data()

        self.ring_selection = []
        step = 5
        pos = 1
        pos_last = 0
        while pos < self.count_nodes:
            self.ring_selection.append([pos_last, min(pos, self.count_nodes)])
            pos_last = pos
            pos += step
            step += 5
        # print(self.ring_selection)
        self.count_rings = len(self.ring_selection)

        self.expands = nn.Parameter(torch.zeros(self.count_rings, dtype=torch.float64))

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
        # if self.init_randomly:
        #     self.expands = nn.Parameter(torch.rand(self.count_nodes, dtype=torch.float64) * 1.2 - 0.6)
        # else:
        #     self.expands = nn.Parameter(torch.zeros(self.count_nodes, dtype=torch.float64))
        # 准备好 boards 数据
        self.boards = self.get_boards().to(self.device)
        self.paddings_raw = self.get_paddings(source=self.position_raw)
        # 到设备
        self.to(self.device)

    # 计算在当前伸缩值状态下，主索节点的位置
    def update_position(self, expand_source=None):
        if expand_source is None:
            self.limit_expands()
            expand_source = self.expands
        else:
            if isinstance(expand_source, np.ndarray):
                expand_source.clip(-0.6, 0.6)
            else:
                expand_source = torch.clamp(expand_source, torch.tensor(-0.6), torch.tensor(0.6))
        # print(self.position_raw, self.unit_vectors, self.expands)
        # print(self.unit_vectors, self.expands)
        # print(self.unit_vectors.shape, self.expands.shape)
        # self.position = self.position_raw + torch.dot(self.unit_vectors,
        #                                               torch.reshape(self.expands, (len(self.expands), 1)))
        # print("Why!")
        expand_temp = torch.zeros(self.count_nodes, device=self.device, dtype=torch.float64)
        for r in range(len(self.ring_selection)):
            ring = self.ring_selection[r]
            expand_temp[ring[0] : ring[1]] = expand_source[r]
        m = self.unit_vectors * expand_temp.reshape((self.count_nodes, 1))
        # m = torch.matmul(self.unit_vectors, self.expands)
        position = self.position_raw + m
        # print(self.position.shape)
        # for i in range(self.count_nodes):
        #     self.position[i] = self.position_raw[i] + self.unit_vectors[i] * self.expands[i]
        return position

    # 判断当前伸缩条件下能否满足伸缩量限制
    def is_expands_legal(self) -> bool:
        r = torch.max(self.expands) <= 0.6 and torch.min(self.expands) >= -0.6
        return r

    # 数一数有多少个伸缩量不满足要求
    def count_illegal_expands(self) -> int:
        # ones = torch.ones(self.expands.shape, device=self.device, dtype=torch.float64)
        # zeros = torch.zeros(self.expands.shape, device=self.device, dtype=torch.float64)
        count_1 = torch.where(-0.6 > self.expands, 1., 0.)
        count_2 = torch.where(0.6 < self.expands, 1., 0.)
        count = count_1 + count_2
        return torch.sum(count)

    # 判断当前伸缩条件能否满足间隔限制
    def is_padding_legal(self, paddings=None) -> bool:
        paddings = self.get_paddings(source=self.position) if paddings is None else paddings
        d = torch.abs(paddings - self.paddings_raw)
        count = torch.where(d > 0.0007 * self.paddings_raw, 1., 0.)
        return count.sum() == 0
        # return torch.sum(torch.tensor([
        #     torch.sum(torch.tensor([
        #         (1 if (torch.abs(paddings[i][j] - self.paddings_raw[i][j]) < 0.0007 * self.paddings_raw[i][j]) else 0)
        #         for j in range(3)
        #     ])) for i in range(len(paddings))
        # ])) == 0

    # 得到间隔误差
    def get_padding_loss(self, weight: float = 1) -> torch.Tensor:
        paddings = self.get_paddings(source=self.position)
        w = torch.tensor(1)
        if self.is_padding_legal(paddings=paddings):
            # return torch.tensor(0)
            w = w * 0.5

        # length = len(paddings)
        # r = torch.sum(torch.from_numpy(
        #     np.array([
        #         torch.sum(torch.tensor([
        #             (paddings[i][j] - self.paddings_raw[i][j]) ** 2 for j in range(3)
        #         ])).numpy() for i in range(length)
        #     ]) ** 2
        # ))
        r = torch.sum(((paddings - self.paddings_raw) ** 2).flatten()) * w
        return r * weight

    # 处理限制伸缩量
    def limit_expands(self):
        # self.expands = nn.Parameter(torch.clamp(self.expands, torch.tensor(-0.6), torch.tensor(0.6)))
        self.expands.data.clamp_(torch.tensor(-0.6), torch.tensor(0.6))

    # 处理限制间隔误差
    def limit_paddings(self):
        orders = [
            [1, 2],
            [0, 2],
            [0, 1]
        ]
        for i in range(len(self.count_triangles) - 1, 0, -1):
            board = self.boards[i]
            distance: torch.Tensor = board.transpose(0, 1)[0] ** 2 + board.transpose(0, 1)[1] ** 2
            # 选定最近的那个点
            select = torch.argmin(distance)
            near = board[select]

    # 得到伸缩误差
    def get_expand_loss(self, weight: float = 1) -> torch.Tensor:
        r = torch.sum(torch.stack(
            [torch.tensor(0).to(self.device) if (-0.6 <= expand <= 0.6) else (torch.abs(expand) - 0.6) for expand in
             self.expands]))
        self.limit_expands()
        # return torch.tensor(0) * weight
        return r * weight

    # 得到光通量误差
    def get_light_loss(self, weight: float = 1) -> torch.Tensor:
        m = torch.as_tensor([0, 0, 1], device=self.device, dtype=torch.float64)
        S = np.pi * FAST.R_CABIN ** 2
        count_surface = 0
        sum_surface = 0
        for board in self.boards:
            if board.transpose(0, 1)[0].max() ** 2 + board.transpose(0, 1)[1].max() ** 2 <= FAST.R_SURFACE ** 2:
                n_plane, D = triangle_to_plane(board)
                dot = torch.dot(n_plane, m)
                theta = torch.arccos(dot / torch.sqrt(torch.sum(n_plane ** 2)))
                # -> 0 ?
                s1 = S * torch.cos(theta * 2)
                count_surface += 1
                sum_surface += s1
            else:
                continue

        return (count_surface * S - sum_surface) * weight

    # 得到拟合精度误差
    def get_fitting_loss(self, weight: float = 1) -> torch.Tensor:
        loss_sum = 0
        count_surface = 0
        # 得到底部顶点
        vertex = self.get_vertex()
        index = 0
        # for board in self.boards:
        for index_node in range(len(self.boards)):
            board = self.boards[index_node]
            # 反射板中心
            center = get_board_center(board)
            # 在半径 150m 以外的板子不计算拟合精度误差
            # a = board.transpose(0, 1)
            # b = board.transpose(0, 1)[0]
            # c = board.transpose(0, 1)[0].max()

            if board.transpose(0, 1)[0].max() ** 2 + board.transpose(0, 1)[1].max() ** 2 <= FAST.R_SURFACE ** 2:
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
                    # 不考虑 A0
                    # if index_node == 0:
                    #     continue
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
                # n_surface, n_board = -normalizing(n_surface), normalizing(n_board)
                # 求点积
                dot = torch.dot(n_surface, n_board)
                loss_sum += dot
                # if index_node == 0:
                #     logger.warning(f"A0: 1 - dot = {1 - dot.clone().detach().item()}")
            else:
                # loss_sum += 1
                # count_surface += 1
                # print(board.transpose(0, 1)[0].max() ** 2 + board.transpose(0, 1)[1].max() ** 2)
                continue

        # logger.warning(f'ALL: 1 -Sigma dot = {1 - (loss_sum / count_surface)}')
        r = (1 - (loss_sum / count_surface)) * weight
        return r

    # 计算整体误差
    def get_loss(self) -> torch.Tensor:
        self.boards = self.get_boards()
        loss_1 = self.get_expand_loss(weight=self.loss_weights[0])
        loss_2 = self.get_padding_loss(weight=self.loss_weights[0])
        loss_3 = self.get_fitting_loss(weight=self.loss_weights[1])
        # loss_3 = torch.tensor(0).to(self.device) * self.loss_weights[1]
        loss_4 = self.get_light_loss(weight=self.loss_weights[2])
        loss_all = [loss_1, loss_2, loss_3, loss_4]
        logger.info(f"loss: {[x.item() for x in loss_all]}")
        loss = sum(loss_all)
        return loss

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
    def get_vertex(self) -> torch.Tensor:
        # board = self.get_bottom_board()
        # if board is None:
        #     raise Exception("取得顶点参考板错误！")
        # for triangle in self.triangles_data:
        #     if torch.eq(torch.as_tensor(self.get_board(triangle)), board).sum().item() == 3:
        #         logger.info(f'using board: {triangle}')
        #         break
        # n, D = triangle_to_plane(board)
        # return -D / n[2]

        # boards_selected = self.boards[:(1 + 5 + 10 + 15 + 20)]
        # length = len(boards_selected)
        # board_ave = torch.sum(boards_selected.transpose(0, 1), dim=1) / length
        # z = get_board_center(board_ave)[2] - 0.8
        # print('-z =', -z, 'R =', FAST.R)
        # return z

        return self.vertex
        # return torch.tensor(-300.4)

    # 整体旋转整个模型
    def rotate(self, alpha: float, beta: float, unit_degree: bool = False):
        if unit_degree:
            alpha = alpha / 360 * np.pi * 2
            beta = beta / 360 * np.pi * 2
        m = get_rotation_matrix(alpha, beta).to(self.device)
        # self.position = torch.mm(self.position, m)
        # self.position_raw = torch.mm(self.position_raw, m)
        # self.actuator_head = torch.mm(self.actuator_head, m)
        # self.actuator_base = torch.mm(self.actuator_base, m)
        # self.unit_vectors = torch.mm(self.unit_vectors, m)
        # print(m.shape, self.position.shape)
        self.position = torch.mm(m, self.position.transpose(0, 1)).transpose(0, 1)
        self.position_raw = torch.mm(m, self.position_raw.transpose(0, 1)).transpose(0, 1)
        self.actuator_head = torch.mm(m, self.actuator_head.transpose(0, 1)).transpose(0, 1)
        self.actuator_base = torch.mm(m, self.actuator_base.transpose(0, 1)).transpose(0, 1)
        self.unit_vectors = torch.mm(m, self.unit_vectors.transpose(0, 1)).transpose(0, 1)

    # 计算主索节点位置
    def forward(self):
        self.position = self.update_position()
        loss = self.get_loss()
        return loss


g_fig = None
g_frame: np.ndarray = None
g_draw_kwargs: dict = None
g_exit: bool = False


# 绘制当前图像
def draw(model: FAST, **kwargs):
    global g_frame, g_draw_kwargs
    if draw_threaded:
        g_frame = model.expands.clone().cpu().detach().numpy()
        g_draw_kwargs = kwargs
    else:
        g_draw_kwargs = kwargs
        draw_thread(source=model.expands.clone().cpu().detach().numpy())

    # frame = model.expands.clone().cpu().detach().numpy()
    # position = model.update_position(expand_source=frame)
    # size = (int(position.transpose(0, 1)[0].max() - position.transpose(0, 1)[0].min() + 1),
    #         int(position.transpose(0, 1)[1].max() - position.transpose(0, 1)[1].min() + 1))
    # im = np.zeros(size, dtype=np.uint8)
    # for p in position:
    #     pos = (int((p[0] - position.transpose(0, 1)[0].min())), int((p[1] - position.transpose(0, 1)[1].min())))
    #     cv2.circle(im, center=pos, radius=5, color=(0xFF - int(0xFF * (p[2] - position.transpose(0, 1)[2].min()) / (
    #             position.transpose(0, 1)[2].max() - position.transpose(0, 1)[2].min()))), thickness=-1)
    #     cv2.imshow('now', im)
    #     cv2.waitKey(1)


def draw_thread(source: torch.Tensor = None):
    global g_frame, g_fig
    while True:
        wait_time: int = 0
        enlarge: float = 500
        if source is None:
            if g_exit:
                return
            if g_frame is None or model_ is None:
                time.sleep(0.05)
                continue
            wait_time: int = g_draw_kwargs.get('wait_time', 0)
            enlarge: float = g_draw_kwargs.get('enlarge', 500)
            if wait_time < 0:
                if g_fig is not None:
                    try:
                        plt.close(g_fig)
                    except Exception as e:
                        print(e)

        # if g_fig is None:
        #     g_fig = plt.figure(1, figsize=(4, 4), dpi=80)

        # fig1 = plt.figure(1, figsize=(4, 4), dpi=80)
        fig1 = plt.figure(dpi=80)

        plt.xlim(-300, 300)
        plt.ylim(-300, 300)

        # ax = plt.subplot(2, 2, 2, projection='3d')
        # plt.sca(ax)
        ax = plt.axes(projection='3d')
        ax.view_init(elev=10., azim=11)
        # ax.view_init(elev=90., azim=0)
        ax.set_zlim(-400, -100)

        # ax2 = plt.axes(projection='3d')
        # ax2.view_init(elev=10., azim=11)
        # # ax2.view_init(elev=90., azim=0)
        # ax2.set_zlim(-400, -100)

        if source is None:
            expands = g_frame * enlarge
            expands_raw = g_frame
            g_frame = None
        else:
            expands = source * enlarge
            expands_raw = source

        fig2 = plt.figure(dpi=80)
        ax2 = plt.axes()
        # ax2 = plt.subplot(2, 2, 1)
        plt.sca(ax2)
        # 画 expands
        plt.plot([i for i in range(len(expands_raw))], expands_raw)

        def draw_it(expands_, c='g'):
            position: torch.Tensor = model_.update_position(expand_source=expands_)
            points = position.clone().cpu().numpy()
            ax.scatter3D(points.T[0], points.T[1], points.T[2], c=c, marker='.')

        draw_it(expands, 'g')
        # draw_it(expands_raw, 'm')

        # ax2.scatter3D(points.T[0], points.T[1], points.T[2], c="g", marker='.')
        # X, Y = np.meshgrid(points.T[0], points.T[1])
        # Z = (1 - X / 2 + X ** 3 + Y ** 4) * np.exp(-X ** 2 - Y ** 2)
        # plt.contourf(X, Y, Z)

        if source is None:
            # if wait_time == 0:
            #     plt.show()
            if 0 > wait_time:
                plt.show()
            if 0 == wait_time:
                plt.draw()
                if wait_time > 0:
                    plt.pause(wait_time)
                else:
                    # plt.show(block=False)
                    plt.pause(wait_time + 0.5)
                # plt.close(g_fig)
            elif wait_time < 0:
                plt.draw()
            plt.clf()
        else:
            # fig = plt.figure(1)
            plt.draw()
            plt.pause(3)
            plt.close(fig1)
            plt.close(fig2)
            break


def main(alpha: float = 0, beta: float = 0, learning_rate: float = 1e-4, show: bool = True, wait_time: int = 0,
         out: str = 'data/附件4.xlsx', module_path: str = None, load_path: str = None, **kwargs):
    global model_, g_exit
    model_ = FAST(**kwargs)
    model = model_
    if load_path is not None:
        try:
            model.load_state_dict(torch.load(load_path))
        except FileNotFoundError:
            logger.error(f"No module path: {load_path}")
    if draw_threaded:
        thread_draw = threading.Thread(target=draw_thread)
        thread_draw.setDaemon(True)
        thread_draw.start()

    # 旋转模型
    # test_rotation(model)
    model.rotate(alpha, beta, unit_degree=True)
    # test_triangle_order(model)
    # test_r2(model)
    # exit()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    try:
        for i in trange(1000):
            optimizer.zero_grad()
            loss = model()
            logger.info(f'epoch {i} loss: {loss.item()}')
            logger.warning(f"vertex: {model.vertex.clone().cpu().detach().item()}")
            if not model.is_expands_legal():
                logger.warning(f'不满足伸缩限制！共{model.count_illegal_expands()}')
            if not model.is_padding_legal():
                logger.warning(f'不满足间隔变化限制！')
            loss.backward()
            optimizer.step()
            print(model.expands)
            if show:
                draw(model, wait_time=wait_time, enlarge=1)
    except KeyboardInterrupt:
        pass
    g_exit = True
    # 进行一个文件的保存
    try:
        logger.info(f'Saving expands data to: {out}')
        writer = pd.ExcelWriter(out, engine='xlsxwriter')
        df = pd.DataFrame({
            '对应主索节点编号': model.name_list,
            '伸缩量（米）': model.expands.cpu().detach().numpy(),
            '': ['' for _ in range(model.count_nodes)],
            '注：至少保留3位小数': ['' for _ in range(model.count_nodes)]
        })
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        worksheet = writer.sheets['Sheet1']
        worksheet.set_column("A:A", 4)
        worksheet.set_column("B:B", 15)
        worksheet.set_column("D:D", 50)
        writer.close()
    except Exception as e:
        logger.error('保存数据文件出错: %s' % str(e))
        traceback.print_exc()
    # 进行一个模型的保存
    try:
        if module_path is not None:
            logger.info(f'Saving module weights to: {module_path}')
            torch.save(model.state_dict(), module_path)
    except Exception as e:
        logger.error('保存模型文件出错: %s' % str(e))
        traceback.print_exc()
    logger.info('ALL DONE')


def test_rotation(model: FAST):
    for beta in range(45, 90, 5):
        model.rotate(0, beta, unit_degree=True)
        draw_thread(model.expands.clone().cpu().detach().numpy())
        # time.sleep(1)
        model.read_data()
    exit()


def test_triangle_order(model: FAST):
    im = np.zeros((500, 500), dtype=np.uint8)
    for i in range(model.count_triangles):
        triangle = model.triangles_data[i]
        board = model.get_board(triangle).cpu().clone().detach().numpy()
        points = np.array((board.T[:2]).T, dtype=np.int32) + 250
        cv2.fillPoly(im, [points], int(200 - i / model.count_triangles * 200) + 50)
        cv2.imshow('triangles', im)
        cv2.waitKey(1)
    cv2.waitKey(0)


def test_r2(model: FAST):
    position_raw = model.position_raw.clone().cpu().detach().numpy()
    step = 1
    pos = 5
    pos_last = 0
    splits = []
    fig = plt.figure(dpi=80)

    while pos < len(position_raw):
        # print(f'[{pos_last} : {pos}]')
        position_selected = position_raw[pos_last:pos]
        # print(len(position_selected))
        # r2 = np.array([np.sum((i - [0, 0, -300.4]) ** 2) for i in position_selected])
        r2 = np.array([i[2] for i in position_selected])
        splits.append(r2.copy())
        plt.plot([i for i in range(pos_last, pos, 1)], r2)
        pos_last = pos
        pos += step
        step += 5
    print('num[r] =', len(splits))

    # r2 = np.array([np.sum((position_raw[i] - [0, 0, -300.4]) ** 2) / (30 * i) for i in range(len(position_raw))])
    # r2 = np.array([(i + 10) / 500 + position_raw[i][2] / (((i + 1))) for i in range(len(position_raw))])

    # plt.plot([i for i in range(len(r2))], r2)
    # plt.plot([i for i in range(len(position_raw))], r2)
    plt.show()


model_: FAST = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', type=float, default=0, help='设置 alpha 角（单位：度）')
    parser.add_argument('-b', '--beta', type=float, default=90, help='设置 beta 角（单位：度）')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='设置学习率')
    parser.add_argument('-r', '--randomly-init', type=bool, default=False, help='设置是否随机初始化参数')
    parser.add_argument('-p', '--optim', type=str, default='Adam', help='设置梯度下降函数')
    parser.add_argument('-d', '--device', type=str, default=None, help='设置 Tensor 计算设备')
    parser.add_argument('-s', '--show', type=bool, default=False, help='设置是否显示训练中图像')
    parser.add_argument('-w', '--wait-time', type=float, default=0, help='设置图像显示等待时间（单位：秒）')
    parser.add_argument('-o', '--out', type=str, default='data/附件4.xlsx', help='设置完成后数据导出文件')
    parser.add_argument('-m', '--module-path', type=str, default='data/module.pth', help='设置模型保存路径')
    # parser.add_argument('-t', '--load-path', type=str, default='data/module.pth', help='设置模型加载路径')
    parser.add_argument('-t', '--load-path', type=str, default=None, help='设置模型加载路径')
    args = parser.parse_args()
    logger.info(f'参数: {args}')
    main(**args.__dict__)
