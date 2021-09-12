import os
import argparse
import pandas as pd
import time
import matplotlib.pyplot as plt
import traceback
import torch.optim as optim
from tqdm import trange
import threading
from utils import *
import cv2
from base_logger import logger
from fast import FAST

draw_threaded: bool = False

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
        # wait_time: int = 0
        # enlarge: float = 500
        wait_time: int = g_draw_kwargs.get('wait_time', 0)
        enlarge: float = g_draw_kwargs.get('enlarge', 500)
        if source is None:
            if g_exit:
                return
            if g_frame is None or model_ is None:
                time.sleep(0.05)
                continue
            # wait_time: int = g_draw_kwargs.get('wait_time', 0)
            # enlarge: float = g_draw_kwargs.get('enlarge', 500)
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
            points = position.clone().detach().cpu().numpy()
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
         out: str = 'data/附件4.xlsx', module_path: str = None, load_path: str = None, enlarge: float = 500,
         mode: str = 'ring', **kwargs):
    global model_, g_exit
    model_ = FAST(**kwargs)
    model = model_
    if load_path is not None:
        try:
            if mode == FAST.MODE_SINGLE:
                path = os.path.join(os.path.dirname(load_path),
                                    f"{os.path.basename(load_path).split('.')[0]}_"
                                    f"{mode}.{load_path.split('.')[-1]}")
            else:
                path = load_path
            try:
                if os.path.exists(path):
                    model.mode = mode
                    model.init_data()
                model.load_state_dict(torch.load(path))
            except FileNotFoundError:
                logger.warning(f'No single module path: {path}, use ring module.')
                model.load_state_dict(torch.load(load_path))
        except FileNotFoundError:
            logger.error(f"No module path: {load_path}")
    if draw_threaded:
        thread_draw = threading.Thread(target=draw_thread)
        thread_draw.setDaemon(True)
        thread_draw.start()

    model.mode = mode
    model.init_data()

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
                # draw(model, wait_time=wait_time, enlarge=100)
                draw(model, wait_time=wait_time, enlarge=enlarge)
    except KeyboardInterrupt:
        pass
    g_exit = True
    # 进行一个文件的保存
    try:
        logger.info(f'Saving expands data to: {out}')
        writer = pd.ExcelWriter(out, engine='xlsxwriter')
        expand_filled = model.get_expand_filled(expand_source=model.expands.cpu().detach()).detach().numpy()
        df = pd.DataFrame({
            '对应主索节点编号': model.name_list,
            '伸缩量（米）': expand_filled,
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
            if model.mode == FAST.MODE_SINGLE:
                path = os.path.join(os.path.dirname(module_path),
                                    f"{os.path.basename(module_path).split('.')[0]}_"
                                    f"{model.mode}.{module_path.split('.')[-1]}")
            else:
                path = module_path
            logger.info(f'Saving module weights to: {path}')
            torch.save(model.state_dict(), path)
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
    weight_default = [5, 2e3, 1e-4]
    parser.add_argument('-w1', '--w1', type=float, default=weight_default[0], help='设置权值1')
    parser.add_argument('-w2', '--w2', type=float, default=weight_default[1], help='设置权值2')
    parser.add_argument('-w3', '--w3', type=float, default=weight_default[2], help='设置权值3')
    parser.add_argument('-e', '--enlarge', type=float, default=500, help='设置图像伸缩放大倍数')
    parser.add_argument('-i', '--mode', type=str, default='ring', help='设置训练模式["ring", "single"]')
    args = parser.parse_args()
    logger.info(f'参数: {args}')
    main(**args.__dict__)
