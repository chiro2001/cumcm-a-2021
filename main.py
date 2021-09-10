# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %% [markdown]
# # cumcm-a
# %% [markdown]
# ## 准备阶段
# 
# ### 数据读取

# %%
import pandas as pd

# 主索节点的坐标和编号
data1 = pd.read_csv("data/附件1.csv", encoding='ANSI')
# print('主索节点的坐标和编号:\n', data1)
nodes_data = {}
for d in data1.itertuples():
  nodes_data[d[1]] = {
    'position': tuple(d[2:]),
    # 伸缩量，即所有需要求解的变量
    'expand': 0
  }

# 促动器下端点（地锚点）坐标、
# 基准态时上端点（顶端）的坐标，
# 以及促动器对应的主索节点编号
data2 = pd.read_csv("data/附件2.csv", encoding='ANSI')
# print('data2:\n', data2)
for d in data2.itertuples():
  nodes_data[d[1]]['actuactor'] = tuple(d[2:5])
  nodes_data[d[1]]['base_state'] = tuple(d[5:8])

# print(nodes_data)

triangles_data = []

# 反射面板对应的主索节点编号
data3 = pd.read_csv("data/附件3.csv", encoding='ANSI')
# print('data3:\n', data3)
for d in data3.itertuples():
  triangles_data.append(tuple(d[1:]))

# print(triangles_data)

# %% [markdown]
# ### 测试绘图
# 
# 绘制点的函数

# %%
import matplotlib.pyplot as plt
import numpy as np

# 绘制当前图像
def draw_points(points: np.ndarray = None, nodes_data_: dict = nodes_data):
  if points is None:
    points = to_points(nodes_data_ = nodes_data_)
  ax = plt.axes(projection='3d')
  ax.scatter3D(points.T[0], points.T[1], points.T[2], c="g", marker='.')
  points2 = to_points(nodes_data_ = nodes_data_, dict_key='base_state')
  ax.scatter3D(points2.T[0], points2.T[1], points2.T[2], c="c", marker='.')
  points3 = to_points(nodes_data_ = nodes_data_, dict_key='actuactor')
  ax.scatter3D(points3.T[0], points3.T[1], points3.T[2], c='m', marker='.')
  plt.show()

# 计算在当前伸缩值状态下，主索节点的位置(position)
def update_expand(nodes_data_: dict = nodes_data):
  pass

def to_points(nodes_data_: dict = nodes_data, dict_key: str = 'position'):
  points = []
  for name in nodes_data_:
    node = nodes_data_[name]
    points.append(node[dict_key])
  return np.array(points)

draw_points()


