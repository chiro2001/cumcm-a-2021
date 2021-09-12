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