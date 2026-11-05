# YOLO26 抓取模型

将自己训练的 `best.pt` 放在本目录。从工作空间根目录启动时，检测节点默认读取
`models/best.pt`，使用第一块 NVIDIA GPU（`yolo_device:=0`）。

训练输出通常位于 `runs/detect/<训练名称>/weights/best.pt`。也可以保留训练目录，
启动时传 `yolo_model:=/绝对路径/best.pt`，无需修改 Python 代码。
默认推理尺寸为 `yolo_imgsz:=640`，置信度阈值为 `yolo_confidence:=0.45`。

使用自己的数据集训练 YOLO26 时，`data.yaml` 中的 `names` 就是模型类别名；例如：

```bash
yolo detect train model=yolo26n.pt data=/绝对路径/data.yaml \
  imgsz=640 epochs=100 device=0 project=runs/detect name=grasp_yolo26
cp runs/detect/grasp_yolo26/weights/best.pt models/best.pt
```

显存不足时降低 `batch` 或 `imgsz`。部署前确认 `best.pt` 的类别名与抓取配置一致，
否则通过 `yolo_class_map` 映射到 `red`、`blue` 或 `yellow`。

模型类别直接来自 `best.pt`。默认 `pick --color auto` 选择置信度最高的目标；
若继续用 `pick --color red` 等旧抓取配置，可传
`yolo_class_map:='{"red_cube":"red","blue_cube":"blue"}'`。这只是类别映射，
不会运行颜色分割。三维位置仍来自 D435 深度和标定后的 TF。

这里不提供占位 `best.pt`；没有训练权重时，默认 YOLO 入口会清楚报告文件缺失。
颜色识别备用通过 `detector_backend:=color` 显式开启。权重文件已被 `.gitignore` 排除。
