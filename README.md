# SpikingNav 仿真复现

这是 [SpikingNav: Robust Embodied Navigation with Spiking Neural Policies](https://arxiv.org/abs/2608.05078) 的 **RoboTHOR 仿真复现**。论文没有官方代码，补充材料 S1 也未公开。本仓库按正文公式重实现 SSE + SPN，并接入 AllenAct / RobustNav 风格的 PointNav、ObjectNav 训练与视觉腐蚀评估。

**不包含** Thruster-V2 神经形态芯片部署。这里的前向是 PyTorch 上的稠密卷积（同一张图循环 `T=4`），墙钟时间主要花在 AI2-THOR 渲染和 PPO 反传上，不能当作芯片推理速度。

## 方法

- **SSE**：SEW-ResNet18 风格脉冲骨干（内部步数 `T=4`）+ 两层卷积压缩（512 → 128 → 32）+ 目标嵌入（32 维）+ 两层融合，输出 `r_t ∈ R^{1568}`。
- **SPN**：`v_t = λ u_{t-1} + W_r r_t + W_h u_{t-1}`，脉冲门控硬复位 `u_t = v_t ⊙ (1 - s_t)`，膜电位限制在 `±10ϑ`，再用线性头输出 actor / critic。
- **ANNNav**：匹配的 ResNet18 + GRU(512) 基线。
- **训练**：AllenAct PPO。`clip=0.1`，`value_loss_coef=0.5`，`entropy_coef=0.01`，`lr=3e-4` 线性衰减，`rollout=128`，`update_repeats=4`，`γ=0.99`，`gae_λ=0.95`。PointNav 7500 万步，ObjectNav 3 亿步。每 500 万步存一次检查点。

未公开超参写在 `spikingnav/config.py`：`hidden=512`，`λ=0.5`，`ϑ=1.0`，代理梯度为 arctan（`α=2`）。骨干用 ImageNet ANN 权重初始化 SEW-ResNet，而不是完整 BuSNN（BuSNN 同样无代码）。循环权重 `W_h` 用较小正交增益初始化，避免 128 步展开时膜电位溢出。

环境：RoboTHOR Locobot，RGB 400×300 再缩到 224，步长 0.25 m，转向 30°，单回合最多 500 步，水平视场 79°。ObjectNav 成功距离（可见性）1.0 m，动作含 `LookUp` / `LookDown`，12 类目标：AlarmClock、Apple、BaseballBat、BasketBall、Bowl、GarbageCan、HousePlant、Laptop、Mug、SprayBottle、Television、Vase。PointNav 成功距离 0.2 m，动作为 MoveAhead / RotateLeft / RotateRight / End。

## 目录

```
main.py                 AllenAct 入口，额外接受 -vc / -vs
spikingnav/             LIF、SSE、SPN、ANNNav、腐蚀、实验配置
projects/               AllenAct 能发现的实验类
scripts/                环境、数据、THOR 下载、训练、评估
tests/                  单测（不启动模拟器）
datasets/               数据下载位置（不进 git）
storage/                检查点、TensorBoard、报错现场（不进 git）
third_party/robustnav   RobustNav 评估集来源（不进 git）
```

## 环境

需要 Linux、已安装的 Miniconda 或 Anaconda、NVIDIA 驱动，以及和驱动主版本一致的 Vulkan（AI2-THOR CloudRendering 用它画图）。环境名是 `spikingnav`，Python 3.9。本机验证过的组合是 Python 3.9.23 与 `torch 2.8.0+cu128`。

还没有 conda 时，先装 Miniconda，装完重新打开终端：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

在仓库根目录创建环境并装上仿真栈。`moviepy` 必须是 `1.0.3`，2.x 会让 AllenAct 的 `from moviepy import editor` 失败。

```bash
cd /path/to/SNN-Nav
conda create -n spikingnav python=3.9 -y
conda activate spikingnav

# 与本机驱动匹配的 CUDA 12.8 轮子。驱动更旧时，把 cu128 换成对应的 CUDA 版本。
pip install torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
pip install -e .
pip install 'ai2thor>=2.7.4' 'allenact' 'allenact_plugins[ithor]' 'moviepy==1.0.3'
pip install tensorboard
```

`requirements.txt` 里的 `torch>=1.12` 有时会把上面的 CUDA 轮子换成 PyPI 上的另一构建。装完后确认版本仍带 `+cu128`，并且能看到 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

若打印出来的不是 `2.8.0+cu128`，把 `pip install torch==2.8.0 ... cu128` 那一行再执行一次。

也可以用脚本一次做完。它按 `environment.yml` 创建 `spikingnav`（已存在则跳过创建），然后执行 `pip install -e .` 以及 AllenAct / AI2-THOR。脚本不会指定 CUDA 轮子，GPU 训练前仍要用上面的命令核对 `torch.__version__`。

```bash
bash scripts/setup_env.sh
conda activate spikingnav
```

以后每次开新终端：

```bash
conda activate spikingnav
cd /path/to/SNN-Nav
```

只检查模型和单测、不启动模拟器时：

```bash
pytest -q
python scripts/smoke_test.py
```

## 数据与渲染器

```bash
bash scripts/download_robothor.sh all
```

会下载 AllenAct 的 RoboTHOR PointNav / ObjectNav 训练集，并浅克隆 [allenai/robustnav](https://github.com/allenai/robustnav)，把其中的 `robustnav_eval` 链接到 `datasets/robothor-{pointnav,objectnav}/robustnav_eval`。只要其中一项时，把 `all` 换成 `robothor-pointnav` 或 `robothor-objectnav`。

首次训练前还要有 AI2-THOR CloudRendering 包（约 800MB，commit `f0825767cd50d69f666c7f282e54abfe58f1e917`）。`scripts/train.sh` 会调用 `scripts/download_thor.sh`，用 `aria2c`（没有则用 `wget -c`）断点续传到 `~/.ai2thor/releases/`。不要等 AllenAct 自己下：它的环境启动超时是 300 秒，慢速下载会被杀掉。

默认无头 CloudRendering，并会清掉 `DISPLAY`。要用本机 X11 时：

```bash
SPIKINGNAV_USE_X11=1 bash scripts/train.sh objectnav spiking smoke
```

多张 GPU 时，NVIDIA Vulkan ICD 有时会让同一张卡出现两次，AI2-THOR 会在 `controller.py` 里断言失败。`scripts/write_cuda_vulkan_map.py` 会写 `~/.ai2thor/cuda-vulkan-mapping.json`，只保留每个 UUID 的第一个 Vulkan 设备。没有 Vulkan 设备的 CUDA 卡不会被拿来训练。

## 训练

在仓库根目录、已 `conda activate spikingnav` 的前提下：

```bash
bash scripts/train.sh <objectnav|pointnav> <spiking|ann> <full|smoke> [seed]
```

`seed` 默认 `12345`。

短程冒烟（1 个进程、128 步、只用 0 号卡），用来确认模拟器和反传能走通：

```bash
bash scripts/train.sh objectnav spiking smoke
```

论文设定的完整训练：

```bash
bash scripts/train.sh objectnav spiking full   # 3 亿步
bash scripts/train.sh pointnav spiking full    # 7500 万步
bash scripts/train.sh objectnav ann full
bash scripts/train.sh pointnav ann full
```

完整模式若没有设置 `CUDA_VISIBLE_DEVICES`，脚本会改用 Vulkan 能对应上的物理卡，并重新编号成 PyTorch 的 `0,1,2,...`。机器上别的任务占着卡时，只绑空闲卡，并按显存把并行环境数降下来，例如单卡 8 个环境：

```bash
CUDA_VISIBLE_DEVICES=0 SPIKINGNAV_GPU_IDS=0 SPIKINGNAV_NUM_PROCESSES=8 \
  bash scripts/train.sh objectnav spiking full
```

8 个环境、1 张 4090、ObjectNav 3 亿步，按目前的实测外推大约是大半年到一年。4 张空闲 4090、默认 60 个环境，粗估 1.5–4 个月。日志里出现 FPS 后，用 `300000000 / FPS / 86400` 估算剩余天数。

### 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `SPIKINGNAV_NUM_PROCESSES` | 60（smoke 为 1） | 并行仿真进程数 |
| `SPIKINGNAV_MAX_STEPS` | ObjectNav 3e8 / PointNav 7.5e7（smoke 为 128） | 总环境步数，所有进程合计 |
| `SPIKINGNAV_GPU_IDS` | 全部可见卡的本地编号 | 交给 AllenAct 的设备号 |
| `CUDA_VISIBLE_DEVICES` | 完整模式自动设成有 Vulkan 的卡 | 本进程能看见的物理卡 |
| `SPIKINGNAV_VIS_CHUNK` | 4 | 视觉编码的梯度检查点块大小，用来避免 24GB 卡上 PPO 反传 OOM。设为 0 则一次前向整批 |
| `SPIKINGNAV_USE_X11` | 0 | 设为 1 时用显示器而不是 CloudRendering |

### 输出位置

一次 ObjectNav Spiking 完整训练的标签是 `rnav_objectnav_spiking_full`，目录名带启动时间，例如 `2026-09-23_02-06-34`。

- 检查点：`storage/objectnav-spiking-rgb/checkpoints/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/exp_...__steps_<步数>.pt`
- TensorBoard：`storage/objectnav-spiking-rgb/tb/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/`
- 本次实际使用的配置副本：`storage/objectnav-spiking-rgb/used_configs/.../<时间戳>/`

PointNav、ANN 只是把路径里的 `objectnav` / `spiking` 换成对应名字。检查点大约每 500 万步写一次，同时在该检查点上跑验证集。训练日志大约每 1 万步记一次 success、SPL、reward、回合长度、PPO loss 和 FPS。

当前环境写事件用的是 `tensorboardX`。要在浏览器里看曲线，需要再装查看器：

```bash
pip install tensorboard
tensorboard --logdir storage/objectnav-spiking-rgb/tb --port 6006
```

### 断点续训

`scripts/train.sh` 不接收检查点参数。500 万步之前中断，没有 `.pt` 可以续。有检查点之后，用和开训时相同的 `CUDA_VISIBLE_DEVICES`、`SPIKINGNAV_GPU_IDS`、`SPIKINGNAV_NUM_PROCESSES`，在 `python main.py` 上加 `-c`：

```bash
CUDA_VISIBLE_DEVICES=0 SPIKINGNAV_GPU_IDS=0 SPIKINGNAV_NUM_PROCESSES=8 \
python main.py \
  -o storage/objectnav-spiking-rgb \
  -b projects/spikingnav_baselines/experiments \
  objectnav_robothor_spiking_ddppo \
  -s 12345 \
  --extra_tag rnav_objectnav_spiking_full \
  -c storage/objectnav-spiking-rgb/checkpoints/Objectnav-RoboTHOR-SpikingNav-RGB-DDPPO/rnav_objectnav_spiking_full/<时间戳>/exp_....pt
```

续训会从检查点里的步数接着走到原定总步数，并新建一个时间戳目录，不覆盖旧日志。文件名以 `error_for_exp_` 开头的是 NaN 现场，里面的权重不可用。

## 评测

干净验证集（`robustnav_eval`）：

```bash
bash scripts/eval.sh objectnav spiking path/to/checkpoint.pt
bash scripts/eval.sh pointnav ann path/to/checkpoint.pt
```

单一腐蚀，严重程度默认 5：

```bash
bash scripts/eval.sh objectnav spiking path/to/checkpoint.pt "Motion Blur" 5
```

七种腐蚀各跑一遍，最后再跑一遍干净集：

```bash
bash scripts/eval_corruptions.sh objectnav spiking path/to/checkpoint.pt 5
```

腐蚀名称与 RobustNav 一致：`Low Lighting`、`Motion Blur`、`Camera Crack`、`Defocus Blur`、`Speckle Noise`、`Lower FOV`、`Spatter`。主指标是 SR 与 SPL。结果写到 `storage/<task>-<model>-eval/`。

`main.py` 在交给 AllenAct 之前吃掉 `-vc` / `-vs`，并写成 `SPIKINGNAV_CORRUPTION`、`SPIKINGNAV_SEVERITY`。实验名是：

- `objectnav_robothor_spiking_ddppo`
- `objectnav_robothor_ann_ddppo`
- `pointnav_robothor_spiking_ddppo`
- `pointnav_robothor_ann_ddppo`

直接调用时要带上仓库根目录的 `PYTHONPATH`，例如：

```bash
export PYTHONPATH="$PWD"
python main.py \
  -o storage/objectnav-spiking-eval \
  -b projects/spikingnav_baselines/experiments \
  objectnav_robothor_spiking_ddppo \
  -c path/to/checkpoint.pt \
  --eval \
  -s 12345 \
  --extra_tag eval_objectnav_spiking \
  -vc "Speckle Noise" -vs 5
```

## 论文对照

| 方法 | Params | FLOPs | PointNav SR/SPL | ObjectNav SR/SPL | 腐蚀均值 SR/SPL |
| --- | --- | --- | --- | --- | --- |
| ANNNav | 14.0M | 4.21G | 98.21 / 82.13 | 31.05 / 14.26 | 8.45 / 3.59 |
| SpikingNav | 12.1M | 0.97G | 96.54 / 72.93 | 34.12 / 16.20 | 13.71 / 5.67 |

完整训练很重，本仓库保证模型与管线可跑通，不自动跑满长训练。数字可能无法逐项对齐：S1 未公开、骨干用 SEW-LIF 而非完整 BuSNN、随机种子与并行数也可能不同。不加载预训练权重时，本地统计的参数量大约是 SpikingNav 12.99M、ANNNav 15.12M。

## 引用

```
@article{zhang2026spikingnav,
  title={SpikingNav: Robust Embodied Navigation with Spiking Neural Policies},
  author={Zhang, Jiahong and Shen, Sijun and Wu, Dehua and Lin, Yifan and Xia, Xuechen and Chu, Xu and Zhang, Youhui and Li, Guoqi},
  journal={arXiv preprint arXiv:2608.05078},
  year={2026}
}
```
