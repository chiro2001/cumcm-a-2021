@echo 看看初始状态

python main.py ^
  --learning-rate 0.0001 ^
  --device cpu ^
  --module-path data/none.pth ^
  --enlarge 500 ^
  --wait-time 0 ^
  --mode ring ^
  --save-only True

@echo 保存第 1 题结果到 vertex.txt

del vertex.txt

python main.py ^
  --learning-rate 0.0001 ^
  --device cpu ^
  --module-path data/temp.pth ^
  --enlarge 500 ^
  --load-path data/p1_re.pth ^
  --wait-time 0.1 ^
  --mode single ^
  --show True ^
  --save-image True ^
  --save-only True

@echo 保存第 2 题结果到 Excel

python main.py ^
  --learning-rate 0.0001 ^
  --device cpu ^
  --module-path data/temp.pth ^
  --enlarge 500 ^
  --load-path data/p2.pth ^
  --wait-time 0.1 ^
  --mode single ^
  --show True ^
  --save-image True ^
  --save-only True ^
  --alpha 36.795 ^
  --beta 78.169

@echo 计算第 3 问数据

python main.py ^
  --learning-rate 0.0001 ^
  --device cpu ^
  --module-path data/temp.pth ^
  --enlarge 500 ^
  --load-path data/p2.pth ^
  --wait-time 0.1 ^
  --mode single ^
  --alpha 36.795 ^
  --beta 78.169 ^
  --calc-only True