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


class FAST(nn.Module):
    # 一些常量设定
    # 半径
    R = 300 + 0.4
    # 口径大小（口径直径）
    D = 250
    F = 0.466
    R_SURFACE = 150
    R_CABIN = 1.
    MODE_RING = 'ring'
    MODE_SINGLE = 'single'

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
        self.position_raw: torch.Tensor = None
        self.position_fixed: torch.Tensor = None
        self.position: torch.Tensor = None
        self.actuator_head: torch.Tensor = None
        self.actuator_base: torch.Tensor = None
        self.triangles_data: list = None
        self.name_list: list = None
        self.name_list_fixed: list = None
        self.index: dict = {}
        self.index_fixed: dict = None
        self.paddings_raw: torch.Tensor = None
        self.unit_vectors: torch.Tensor = None
        self.unit_vectors_fixed: torch.Tensor = None
        self.boards: torch.Tensor = None

        self.nodes_data: dict = None

        self.mode = FAST.MODE_RING

        # self.loss_weights: torch.Tensor = torch.tensor([1e-1, 2e4, 1e-3]).to(self.device)
        # self.loss_weights: torch.Tensor = torch.tensor([5, 2e3, 1e-4]).to(self.device)
        self.loss_weights: torch.Tensor = torch.tensor([
            kwargs.get('w1', 5),
            kwargs.get('w2', 2e3),
            kwargs.get('w3', 1e-4)
        ]).to(self.device)
        logger.info(f'set weight to: {self.loss_weights.clone().detach().numpy()}')
        # self.loss_weights: torch.Tensor = torch.tensor([5, 2e3, 1e-9]).to(self.device)
        self.vertex: nn.Parameter = nn.Parameter(torch.tensor(-FAST.R)).to(self.device)

        self.ring_selection: list = None
        self.count_rings: int = None

        self.read_data()

        self.init_data()

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

    def init_data(self):
        # 主索节点名称到下标号的映射
        self.name_list = list(self.nodes_data.keys()) if self.name_list is None else self.name_list
        if self.name_list_fixed is None:
            self.name_list_fixed = copy.deepcopy(self.name_list)
        self.index = {self.name_list[i]: i for i in range(len(self.name_list))}
        if self.index_fixed is None:
            self.index_fixed = copy.deepcopy(self.index)
        self.position_raw = torch.from_numpy(
            np.array([self.nodes_data[name]['position_raw'] for name in self.name_list])).to(self.device)
        # print([self.nodes_data[name]['position'] for name in self.name_list])
        self.position = torch.from_numpy(np.array([self.nodes_data[name]['position'] for name in self.name_list])).to(
            self.device)
        if self.position_fixed is None:
            self.position_fixed = self.position_raw.clone()
        else:
            if 'position_fixed' in self.nodes_data[self.name_list[0]]:
                self.position_fixed = torch.from_numpy(
                    np.array([self.nodes_data[name]['position_fixed'] for name in self.name_list])).to(self.device)
        self.actuator_head = torch.from_numpy(
            np.array([self.nodes_data[name]['actuator_head'] for name in self.name_list])).to(self.device)
        self.actuator_base = torch.from_numpy(
            np.array([self.nodes_data[name]['actuator_base'] for name in self.name_list])).to(self.device)

        # self.unit_vectors = torch.from_numpy(np.array(
        #     [get_unit_vector(self.position_raw[i], self.actuator_base[i]).cpu().numpy() for i in
        #      range(self.count_nodes)]).T).to(self.device)
        self.unit_vectors = torch.from_numpy(np.array(
            [get_unit_vector(self.position_raw[i], self.actuator_base[i]).cpu().numpy() for i in
             range(self.count_nodes)])).to(self.device)
        if self.unit_vectors_fixed is None:
            self.unit_vectors_fixed = self.unit_vectors.clone()
        else:
            if 'unit_vectors_fixed' in self.nodes_data[self.name_list[0]]:
                self.unit_vectors_fixed = torch.from_numpy(
                    np.array([self.nodes_data[name]['unit_vectors_fixed'] for name in self.name_list])).to(self.device)
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
        if self.mode == FAST.MODE_RING:
            if self.expands is None:
                self.expands = nn.Parameter(torch.zeros(self.count_rings, dtype=torch.float64))
        elif self.mode == FAST.MODE_SINGLE:
            if self.expands is None:
                self.expands = nn.Parameter(torch.zeros(self.count_nodes, dtype=torch.float64))
            else:
                self.expands = nn.Parameter(self.get_expand_filled(expand_source=self.expands))

    def sort_z(self):
        logger.debug(f'name_list before: {self.name_list[:5]}...')
        list.sort(self.name_list, key=lambda i: self.nodes_data[i]['position_raw'][2])
        logger.debug(f'name_list after: {self.name_list[:5]}...')
        self.init_data()

    def read_data(self):
        # 主索节点的坐标和编号
        data1 = pd.read_csv("data/附件1.csv", encoding='ANSI')
        # print('主索节点的坐标和编号:\n', data1)
        self.nodes_data = {}
        for d in data1.itertuples():
            self.nodes_data[d[1]] = {
                # 'position': tuple(d[2:]),
                'position_raw': d[2:],
                'position': d[2:]
            }

        # 促动器下端点（地锚点）坐标、
        # 基准态时上端点（顶端）的坐标，
        # 以及促动器对应的主索节点编号
        data2 = pd.read_csv("data/附件2.csv", encoding='ANSI')
        # print('data2:\n', data2)
        for d in data2.itertuples():
            self.nodes_data[d[1]]['actuator_head'] = d[2:5]
            self.nodes_data[d[1]]['actuator_base'] = d[5:8]

        # print(nodes_data)

        triangles_data = []
        # 反射面板对应的主索节点编号
        data3 = pd.read_csv("data/附件3.csv", encoding='ANSI')
        # print('data3:\n', data3)
        for d in data3.itertuples():
            triangles_data.append(tuple(d[1:]))
        # print(triangles_data)

        self.triangles_data = triangles_data
        self.count_triangles = len(self.triangles_data)
        self.count_nodes = len(list(self.nodes_data.keys()))

    def get_expand_filled(self, expand_source: torch.Tensor) -> torch.Tensor:
        expand_filled = torch.zeros(self.count_nodes, device=self.device, dtype=torch.float64)
        for r in range(len(self.ring_selection)):
            if self.mode == FAST.MODE_SINGLE:
                ring = self.ring_selection[r]
                if len(self.expands) == self.count_nodes:
                    return self.expands
                d: torch.Tensor = expand_source[r]
                for i in range(ring[0], ring[1]):
                    expand_filled[i] = d.clone()
                # expand_filled[ring[0]: ring[1]] = torch.stack([d.clone() for _ in range(ring[0], ring[1])])
            else:
                ring = self.ring_selection[r]
                expand_filled[ring[0]: ring[1]] = expand_source[r]
        return expand_filled

    # 计算在当前伸缩值状态下，主索节点的位置
    def update_position(self, expand_source=None, position_raw_source: torch.Tensor = None,
                        unit_vector_source: torch.Tensor = None, enlarge: float = 1):
        if expand_source is None:
            self.limit_expands()
            expand_source = self.expands
        else:
            if isinstance(expand_source, np.ndarray):
                expand_source.clip(-0.6, 0.6)
                expand_source = torch.as_tensor(expand_source)
            else:
                expand_source = torch.clamp(expand_source, torch.tensor(-0.6), torch.tensor(0.6))
        if position_raw_source is None:
            position_raw_source = self.position_raw
        if unit_vector_source is None:
            unit_vector_source = self.unit_vectors
        # print(self.position_raw, self.unit_vectors, self.expands)
        # print(self.unit_vectors, self.expands)
        # print(self.unit_vectors.shape, self.expands.shape)
        # self.position = self.position_raw + torch.dot(self.unit_vectors,
        #                                               torch.reshape(self.expands, (len(self.expands), 1)))
        # print("Why!")
        expand_filled = self.get_expand_filled(expand_source=expand_source)
        m = unit_vector_source * enlarge * expand_filled.reshape((self.count_nodes, 1))
        # m = torch.matmul(self.unit_vectors, self.expands)
        position = position_raw_source + m
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
            if self.mode == FAST.MODE_RING:
                return torch.tensor(0)
            else:
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

    # # 处理限制间隔误差
    # def limit_paddings(self):
    #     orders = [
    #         [1, 2],
    #         [0, 2],
    #         [0, 1]
    #     ]
    #     for i in range(len(self.count_triangles) - 1, 0, -1):
    #         board = self.boards[i]
    #         distance: torch.Tensor = board.transpose(0, 1)[0] ** 2 + board.transpose(0, 1)[1] ** 2
    #         # 选定最近的那个点
    #         select = torch.argmin(distance)
    #         near = board[select]

    # 得到伸缩误差
    def get_expand_loss(self, weight: float = 1) -> torch.Tensor:
        r = torch.sum(torch.stack(
            [torch.tensor(0).to(self.device) if (-0.6 <= expand <= 0.6) else (torch.abs(expand) - 0.6) for expand in
             self.expands]))
        self.limit_expands()
        # return torch.tensor(0) * weight
        return r * weight

    # 得到光通量误差
    def get_light_loss(self, weight: float = 1, get_raw_square: bool = False,
                       get_raw_surface: bool = False) -> torch.Tensor:
        m = torch.as_tensor([0, 0, 1], device=self.device, dtype=torch.float64)
        S = np.pi * FAST.R_CABIN ** 2
        count_surface = 0
        sum_surface = 0
        if get_raw_square:
            # 按照球面标准计算，需要排除无法照到馈源舱的反射板
            # 馈源舱相对于一块板子的可反射小角度
            delta_theta = np.abs(np.arctan(FAST.R_CABIN / (FAST.F * FAST.R)))
            for board in self.boards:
                # 判断板子是否反射到馈源舱
                n_panel, D = triangle_to_plane(board)
                center = get_board_center(board)
                theta = torch.arccos(
                    torch.abs(torch.dot(n_panel, torch.as_tensor([0, 0, 1]) / torch.sqrt(torch.sum(n_panel ** 2)))))
                # theta = np.arccos(np.abs(np.cross(n_panel, [0, 0, 1]) / np.sqrt(np.sum(n_panel ** 2))))
                n_l1 = center - torch.as_tensor([0, 0, -(1 - FAST.F) * FAST.R])
                theta2 = torch.arccos(torch.abs(torch.dot(n_panel, n_l1) / (
                            torch.sqrt(torch.sum(n_l1 ** 2)) * torch.sqrt(torch.sum(n_panel ** 2)))))
                if not theta - delta_theta <= theta2 <= theta + delta_theta:
                    continue
                s1 = S * torch.cos(theta * 2)
                count_surface += 1
                sum_surface += s1
            return sum_surface
        else:
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
            if get_raw_surface:
                return sum_surface
            else:
                return (count_surface * S - sum_surface - 1000) * weight

    # 得到拟合精度误差
    def get_fitting_loss(self, weight: float = 1) -> torch.Tensor:
        loss_sum = 0
        count_surface = 0
        # 得到底部顶点
        vertex = self.get_vertex()
        logger.warning(f'f = {torch.abs(vertex) - (1 - FAST.F) * FAST.R}, z = x**2 / (4 * f) + y**2 / (4 * f) + vertex')
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
        if self.mode == FAST.MODE_RING:
            loss_3 = self.get_fitting_loss(weight=self.loss_weights[1])
            loss_4 = self.get_light_loss(weight=self.loss_weights[2])
        else:
            # loss_3_ = self.get_fitting_loss(weight=self.loss_weights[1])
            # loss_4_ = self.get_light_loss(weight=self.loss_weights[2])
            # logger.info(f"extra loss: {[loss_3_.item(), loss_4_.item()]}")
            loss_3 = torch.tensor(0).to(self.device) * self.loss_weights[1]
            loss_4 = torch.tensor(0).to(self.device) * self.loss_weights[2]
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
        for i in range(len(self.name_list)):
            name = self.name_list[i]
            self.nodes_data[name] = {
                'position': self.position[i].cpu().clone().detach().numpy(),
                'position_raw': self.position_raw[i].cpu().clone().detach().numpy(),
                'position_fixed': self.position_fixed[i].cpu().clone().detach().numpy(),
                'unit_vector_fixed': self.unit_vectors_fixed[i].cpu().clone().detach().numpy(),
                'actuator_head': self.actuator_head[i].cpu().clone().detach().numpy(),
                'actuator_base': self.actuator_base[i].cpu().clone().detach().numpy()
            }
            self.index[name] = i

        self.sort_z()

    # 计算主索节点位置
    def forward(self):
        self.position = self.update_position()
        loss = self.get_loss()
        return loss
