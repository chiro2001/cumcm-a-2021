# cumcm-a

CUMCM Problem A

## Usage

```
usage: main.py [-h] [-a ALPHA] [-b BETA] [-l LEARNING_RATE] [-r RANDOMLY_INIT] [-p OPTIM] [-d DEVICE] [-s SHOW]
               [-g SAVE_IMAGE] [-y SAVE_ONLY] [-w WAIT_TIME] [-o OUT] [-m MODULE_PATH] [-t LOAD_PATH] [-w1 W1] [-w2 W2]
               [-w3 W3] [-e ENLARGE] [-i MODE] [-c CALC_ONLY]

optional arguments:
  -h, --help            show this help message and exit
  -a ALPHA, --alpha ALPHA
                        设置 alpha 角（单位：度）
  -b BETA, --beta BETA  设置 beta 角（单位：度）
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        设置学习率
  -r RANDOMLY_INIT, --randomly-init RANDOMLY_INIT
                        设置是否随机初始化参数
  -p OPTIM, --optim OPTIM
                        设置梯度下降函数
  -d DEVICE, --device DEVICE
                        设置 Tensor 计算设备
  -s SHOW, --show SHOW  设置是否显示训练中图像
  -g SAVE_IMAGE, --save-image SAVE_IMAGE
                        设置是否保存图像数据
  -y SAVE_ONLY, --save-only SAVE_ONLY
                        设置只保存数据不训练
  -w WAIT_TIME, --wait-time WAIT_TIME
                        设置图像显示等待时间（单位：秒）
  -o OUT, --out OUT     设置完成后数据导出文件
  -m MODULE_PATH, --module-path MODULE_PATH
                        设置模型保存路径
  -t LOAD_PATH, --load-path LOAD_PATH
                        设置模型加载路径
  -w1 W1, --w1 W1       设置权值1
  -w2 W2, --w2 W2       设置权值2
  -w3 W3, --w3 W3       设置权值3
  -e ENLARGE, --enlarge ENLARGE
                        设置图像伸缩放大倍数
  -i MODE, --mode MODE  设置训练模式["ring", "single"]
  -c CALC_ONLY, --calc-only CALC_ONLY
                        设置计算第 (3) 问后退出
```

使用 Windows 命令行运行：`run.cmd`。

可能的输出如下：

```
(cumcm) D:\Programs\cumcm-r2>run
看看初始状态

(cumcm) D:\Programs\cumcm-r2>python main.py   --learning-rate 0.0001   --device cpu   --module-path data/none.pth   --enlarge 500   --wait-time 0   --mode ring   --save-only True
[main    ][<module>  ][INFO    ] 参数: Namespace(alpha=0, beta=90, learning_rate=0.0001, randomly_init=False, optim='Adam', device='cpu', show=False, save_image=True, save_only=True, wait_time=0.0, out='data/result.xlsx', module_path='data/none.pth', load_path=None, w1=5, w2=2000.0, w3=0.0001, enlarge=500.0, mode='ring', calc_only=False)
[fast    ][__init__  ][INFO    ] set weight to: [5.e+00 2.e+03 1.e-04]
[fast    ][sort_z    ][DEBUG   ] name_list before: ['A0', 'B1', 'C1', 'D1', 'E1']...
[fast    ][sort_z    ][DEBUG   ] name_list after: ['A0', 'B1', 'C1', 'D1', 'E1']...
  0%|                                                                                                                                                                                                        | 0/1000 [00:00<?, ?it/s]
  [fast    ][get_fitting_loss][WARNING ] f = 139.98638916015625, z = x**2 / (4 * f) + y**2 / (4 * f) + vertex
[fast    ][get_loss  ][INFO    ] loss: [0.0, 0, 0.18458323670733634, 0.01259600640767914]
[main    ][main      ][INFO    ] epoch 0 loss: 0.19717924311501547
[main    ][main      ][WARNING ] vertex: -300.3999938964844
Parameter containing:
tensor([ 9.9999e-05, -1.0000e-04, -1.0000e-04, -1.0000e-04, -1.0000e-04,
        -1.0000e-04, -1.0000e-04, -1.0000e-04,  9.9999e-05,  1.0000e-04,
         1.0000e-04,  1.0000e-04,  1.0000e-04,  1.0000e-04,  1.0000e-04,
         1.0000e-04, -1.0000e-04, -1.0000e-04, -1.0000e-04,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
       dtype=torch.float64, requires_grad=True)
  0%|                                                                                                                                                                                                        | 0/1000 [00:09<?, ?it/s]
[main    ][main      ][WARNING ] trying to save data...
[main    ][main      ][INFO    ] Saving expands data to: data/result.xlsx
[main    ][main      ][WARNING ] vertex loaded from data/vertex.txt.
[main    ][main      ][INFO    ] Saving module weights to: data/none.pth
[main    ][<module>  ][INFO    ] === [ALL DONE] ===
保存第 1 题结果到 vertex.txt

(cumcm) D:\Programs\cumcm-r2>del data\vertex.txt

(cumcm) D:\Programs\cumcm-r2>python main.py   --learning-rate 0.0001   --device cpu   --module-path data/temp.pth   --enlarge 500   --load-path data/p1_re.pth   --wait-time 0.1   --mode single   --show True   --save-image True   --save-only True
[main    ][<module>  ][INFO    ] 参数: Namespace(alpha=0, beta=90, learning_rate=0.0001, randomly_init=False, optim='Adam', device='cpu', show=True, save_image=True, save_only=True, wait_time=0.1, out='data/result.xlsx', module_path='data/temp.pth', load_path='data/p1_re.pth', w1=5, w2=2000.0, w3=0.0001, enlarge=500.0, mode='single', calc_only=False)
[fast    ][__init__  ][INFO    ] set weight to: [5.e+00 2.e+03 1.e-04]
[fast    ][sort_z    ][DEBUG   ] name_list before: ['A0', 'B1', 'C1', 'D1', 'E1']...
[fast    ][sort_z    ][DEBUG   ] name_list after: ['A0', 'B1', 'C1', 'D1', 'E1']...
  0%|                                                                                                                                                                                                        | 0/1000 [00:00<?, ?it/s]
  [fast    ][get_loss  ][INFO    ] loss: [0.0, 0.007952207527706538, 0.0, 0.0]
[main    ][main      ][INFO    ] epoch 0 loss: 0.007952207527706538
[main    ][main      ][WARNING ] vertex: -299.2287902832031
Parameter containing:
tensor([-0.0085, -0.0250, -0.0250,  ...,  0.0002,  0.0002,  0.0002],
       dtype=torch.float64, requires_grad=True)
[main    ][draw_thread][WARNING ] saving images to pics/p1/single_x500_fixed.png, pics/p1/single_expands.png
  0%|                                                                                                                                                                                                        | 0/1000 [00:07<?, ?it/s]
[main    ][main      ][WARNING ] trying to save data...
[main    ][main      ][INFO    ] Saving expands data to: data/result.xlsx
[main    ][main      ][WARNING ] vertex saved to data/vertex.txt.
[main    ][main      ][INFO    ] Saving module weights to: data\temp_single.pth
[main    ][<module>  ][INFO    ] === [ALL DONE] ===
保存第 2 题结果到 Excel

(cumcm) D:\Programs\cumcm-r2>python main.py   --learning-rate 0.0001   --device cpu   --module-path data/temp.pth   --enlarge 500   --load-path data/p2.pth   --wait-time 0.1   --mode single   --show True   --save-image True   --save-only True   --alpha 36.795   --beta 78.169
[main    ][<module>  ][INFO    ] 参数: Namespace(alpha=36.795, beta=78.169, learning_rate=0.0001, randomly_init=False, optim='Adam', device='cpu', show=True, save_image=True, save_only=True, wait_time=0.1, out='data/result.xlsx', module_path='data/temp.pth', load_path='data/p2.pth', w1=5, w2=2000.0, w3=0.0001, enlarge=500.0, mode='single', calc_only=False)
[fast    ][__init__  ][INFO    ] set weight to: [5.e+00 2.e+03 1.e-04]
[fast    ][sort_z    ][DEBUG   ] name_list before: ['A0', 'B1', 'C1', 'D1', 'E1']...
[fast    ][sort_z    ][DEBUG   ] name_list after: ['D27', 'D35', 'D20', 'D34', 'D21']...
  0%|                                                                                                                                                                                                        | 0/1000 [00:00<?, ?it/s]
[fast    ][get_loss  ][INFO    ] loss: [0.0, 0.006592321062528047, 0.0, 0.0]
[main    ][main      ][INFO    ] epoch 0 loss: 0.006592321062528047
[main    ][main      ][WARNING ] vertex: -300.3363342285156
Parameter containing:
tensor([-2.4983e-02, -2.4318e-02, -2.4315e-02,  ..., -5.8094e-05,
        -4.1635e-05,  4.5596e-05], dtype=torch.float64, requires_grad=True)
[main    ][draw_thread][WARNING ] saving images to pics/p2/single_x500_fixed.png, pics/p2/single_expands.png
  0%|                                                                                                                                                                                                        | 0/1000 [00:07<?, ?it/s]
[main    ][main      ][WARNING ] trying to save data...
[main    ][main      ][INFO    ] Saving expands data to: data/result.xlsx
[main    ][main      ][WARNING ] vertex loaded from data/vertex.txt.
[main    ][main      ][INFO    ] Saving module weights to: data\temp_single.pth
[main    ][<module>  ][INFO    ] === [ALL DONE] ===
计算第 3 问数据

(cumcm) D:\Programs\cumcm-r2>python main.py   --learning-rate 0.0001   --device cpu   --module-path data/temp.pth   --enlarge 500   --load-path data/p2.pth   --wait-time 0.1   --mode single   --alpha 36.795   --beta 78.169   --calc-only True
[main    ][<module>  ][INFO    ] 参数: Namespace(alpha=36.795, beta=78.169, learning_rate=0.0001, randomly_init=False, optim='Adam', device='cpu', show=False, save_image=True, save_only=False, wait_time=0.1, out='data/result.xlsx', module_path='data/temp.pth', load_path='data/p2.pth', w1=5, w2=2000.0, w3=0.0001, enlarge=500.0, mode='single', calc_only=True)
[fast    ][__init__  ][INFO    ] set weight to: [5.e+00 2.e+03 1.e-04]
[fast    ][sort_z    ][DEBUG   ] name_list before: ['A0', 'B1', 'C1', 'D1', 'E1']...
[fast    ][sort_z    ][DEBUG   ] name_list after: ['D27', 'D35', 'D20', 'D34', 'D21']...
[fast    ][get_loss  ][INFO    ] loss: [0.0, 0.006592321062528047, 0.0, 0.0]
[fast    ][__init__  ][INFO    ] set weight to: [5.e+00 2.e+03 1.e-04]
调节后馈源舱的接收比: 0.04579877189620465
基准反射球面的接收比: 0.013585394593262586
[main    ][calc      ][WARNING ] Saving p3.txt...
[main    ][<module>  ][INFO    ] === [ALL DONE] ===

```

**目录结构说明：**

```
D:.                                  
│  .gitignore                        
│  base_logger.py                    # 日志
│  fast.py                           # 建模结构
│  LICENSE                           
│  main.py                           # 主程序
│  README.md                         
│  requirements.txt                  # 依赖列表
│  run.cmd                           # 运行脚本
│  utils.py                          # 工具类
│                                    
├─data                               
│      p3.txt                        # 第三题的结果
│      result.xlsx                   # 第二题的结果
│      vertex.txt                    # 第一题的顶点结果
│                                    
└─pics                              # 论文用图像 
```

请查看`data/result.xlsx`以查看第二题得出的数据。