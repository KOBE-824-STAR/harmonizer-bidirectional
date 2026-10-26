# S/M 信息注入方案与代码核查（2026-10-21）

## 当前实现的结论

本目录原始 `double_dit/model.py` 使用**双向门控 cross-attention 的残差注入**，不是提问列出的四种替代方案中的任何一种。主路 S 保持完整空间分辨率和 T1/T5 历史；M 只编码当前图像，空间 token 高宽各池化一半。

零基 block `[3,7,11,15,19]` 的顺序是：

1. `view_exchange`：M 按标定视锥重叠选取其他相机，计算 `m + gated_attention(m, other_views)`。
2. `read_m`：同相机 S 当前时刻 token 以 M 为 K/V，计算 `S + ds`。
3. 前四个交流点还有 `read_s`：M 读取池化后的 S 当前时刻，计算 `M + dm`。`ds`、`dm` 都基于本轮写回前的 S 和跨相机更新后的 m。第 19 层不回写 M。

`GatedAttention` 的门是 `tanh(gate)`，初始为 0。新模块是 1280→320→1280 投影、8 头。两路骨干各 20 层、宽 1280。

已有的相机内外参、视锥重叠邻接矩阵、Plücker 射线编码只提供几何先验；没有深度反投影、逐点对应与遮挡判断，因此不能称为“几何对齐后读取”。M 有六个相机身份嵌入并支持 1–6 个视角；原训练文档实际记录的是双视角训练，不代表已完成完整六视角训练。

## 新增代码与入口

原始模型、训练脚本、启动器、配置、运行记录与 `runs/train/latest.pt` 保持原字节。新代码全部采用独立文件：

- `double_dit/injection.py`：联合注意力、空间调制、几何投影和读取、配置校验。
- `double_dit/model_injection.py`：继承原模型，接入可选方案。
- `double_dit/injection_data.py`：当前合成图像的深度加载与来源校验。
- `scripts/train_injection.py`：独立训练入口，保留原训练损失、T1/T5 生成历史、DDP 和完整 checkpoint。
- `scripts/run_injection.py`：模式隔离的启动入口，运行目录为 `runs/injection_<mode>/<phase>/`。
- `scripts/check_injection_runtime.py`：只读检查迁移后的依赖和数据路径。
- `tests/test_injection.py`：CPU 数学、梯度、遮挡、T1/T5 分支接线与重计算测试。

每种方案均有完整配置 `configs/injection_<mode>.json`，训练尺寸仍是 H×W=`288×512`。

### bidirectional：原始方案对照

调用原始 forward，保留全部原始参数键，不增加新的可学习模块。用于新实验目录内的基线对照。历史检查点仍须通过原脚本及原配置使用；新训练脚本有自己的实现 hash，不能将历史 optimizer resume 当作跨方案迁移。

### one_way：单向读取

M 的跨相机交流保留，S 读取 M；`main_to_condition_after_blocks=[]`，不实例化 `read_s`。删除的是前向的 S→M 特征路径，**不是**阻断 S 最终损失到 M 的反向梯度。M 仍由同一主损失训练。零门控时维持原始单路输出。

### joint：骨干内部联合注意力

在指定 block 内，把两路各自归一化、投影和 RoPE 后的 Q/K/V 拼接，做同一个联合 attention softmax，再拆回两路；复用各自的输出投影、时间 AdaLN 门、文本 cross-attention 和 MLP。删除 `read_m/read_s`，不是在两个完整 block 后再外挂一对 cross-attention 残差。

保留 Transformer 本身的标准残差连接。M 的跨相机交流位于选定联合 block **之前**；因此其调度与原始方案略有区别。跨流范围限定为同一相机：S 当前 token 可读取 S 全历史及 M；S 历史 token 只读 S；M 读取 S 当前 token 和 M。历史 token 可在后续原生 S 自注意力中间接受到当前 token 影响，这是正常时序传播。

六相机不拼成一个全局长序列。M 仍保持 1/4 空间 token 数；`query_chunk_size=128` 分块处理 query，降低单次计算的临时规模。训练总注意力/反向开销仍需真实硬件测量，不能据此承诺固定峰值显存或更快训练。该模式使用骨干原生头数，不使用原 `projection_dim=320/heads=8` 作为联合注意力宽度（它们只用于 M 的 view_exchange）。

**joint 初始不等价于官方单路输出**：新的 K/V 改变 softmax 归一化。其测试单独记录初始差异，不套用零门控等价断言。双向更新也发生于最后一个选中层，不过最后 M 输出不单独用于输出头。

### modulation：空间特征调制

M 经该层与跨相机交流后，生成逐空间位置的 scale/shift，并上采样到 S 当前 token 网格，作用于 S 该层已经完成时间条件化的 self-attention 归一化特征：`normalized_current*(1+scale_M)+shift_M`。没有 M→S 的附加特征残差头，也没有 S→M 读取。

最终 scale/shift 投影零初始化，保持原始单路起点；不将 M 压成一个全局向量，保留局部变化。这里是受自适应归一化启发的空间调制实现，并不声称逐项复现 DiT 论文的 adaLN-Zero 结构。

### geometry：深度对齐的跨相机读取

S 当前 token 使用其相机的当前合成图像深度反投影，通过共同参考坐标系的相机位姿投影到候选源相机，再采样 M 特征。对源相机做图像边界、正深度、有限数值和源深度一致性检查：

`abs(projected_z-source_depth) <= 0.5m + 0.05*abs(projected_z)`。

这些是可配置的初始实验阈值，不是经本数据验证的最优设置。每个 query 只在对齐后的其他相机特征之间做注意力。无对应点时精确零增量，即使输出投影已有非零 bias 也不注入。缺深度直接报错，不回退为重叠注意力。

本模式**停用原未对齐的 M→M view_exchange**，避免被遮挡的源相机先混入另一相机 M 后绕过遮挡掩码。几何跨相机交流发生于 S 对其他相机 M 的读取。M 的本相机 Transformer 仍含空间上下文，因此深度筛选约束的是采样位置及可见性，不保证特征中只含局部单个物体的信息。

深度必须描述当前合成输入，包含插入资产，且推理时可获得；不能用监督目标图像产生深度。相机位姿为 `T_reference_from_cam`，K 与当前输入的像素坐标系匹配。深度值是相机 z 轴米制深度，不是射线长度。像素/token 中心约定沿用原 rays 的 `(index+0.5)*image_size/grid_size`，`grid_sample(align_corners=False)`。

每条 `frame.views[]` 在现有字段旁增加：

```json
{
  "geometry_depth": {
    "path": "/absolute/path/current_composite_depth.npy",
    "sha256": "完整文件的SHA256",
    "kind": "camera_z_m",
    "source": "current_composite",
    "source_image_sha256": "与该条source_sha256一致",
    "available_at_inference": true,
    "image_hw": [288, 512]
  }
}
```

`.npy` 必须是二维浮点数组，与对应 source 图像有相同像素网格和视野；加载器只作最近邻缩放。0/负值/非有限深度表示无效点，不插值填造可见性。校验图像及深度的 hash。元数据声明不能替代真实的深度来源审计。

## 操作命令

在服务器进入 `/data/juziwei/hlk/double_DiT` 后，可以立即执行独立测试和准备配置（不加载大模型、不启动训练）：

```bash
/home/tanhe/miniconda3/envs/recondrive/bin/python -m unittest discover -s tests -p 'test_injection.py' -v
python3 scripts/check_injection_runtime.py --mode geometry
python3 scripts/run_injection.py smoke --mode one_way --prepare-only
python3 scripts/run_injection.py smoke --mode joint --prepare-only
python3 scripts/run_injection.py smoke --mode modulation --prepare-only
python3 scripts/run_injection.py smoke --mode geometry --prepare-only
```

`check_injection_runtime.py` 缺依赖时返回退出码 2；`--prepare-only` 只生成 `prepared_config.json`，不伪装为运行成功，也不覆盖正式的 `config.json`。如果给定 `--manifest`，为该次运行使用独立数据清单。

完整运行环境恢复后，各方案按相同顺序执行，例如：

```bash
python3 scripts/run_injection.py smoke --mode one_way
python3 scripts/run_injection.py native --mode one_way
python3 scripts/run_injection.py ddp_smoke --mode one_way
python3 scripts/run_injection.py train --mode one_way --steps 1000
```

将 `one_way` 换为其他模式即可。geometry 须在每次调用中给同一个有深度的 `--manifest`。正式训练要求该模式自身的 native/DDP 验证、代码 hash、数据配置、交流配置和分辨率一致。不要加载另一模式的 optimizer 状态继续训练；新结构应从官方初始化独立训练再做公平对比。

## 已验证与未验证的界限

独立 CPU 测试覆盖：分块联合注意力与显式稠密掩码的数值一致性；两向信息作用和梯度；无跨相机 batch 泄漏；调制零初始化等价和空间响应；投影坐标、平移、相机后方与遮挡剔除；无对应点零增量；相机置乱等价；深度来源限制；五种模式实际生产 forward 在 T1/T5 下的运行与 checkpoint 重计算梯度一致性。

这些测试使用微型骨干/VAE 替身，执行真正的新 forward 接线。它们**不是**完整 Cosmos 权重、CUDA/Transformer Engine、DDP 或训练质量验证。真实 Cosmos block 的 API/数值兼容仍需恢复外部 runtime 后通过 native 验证。新增 smoke 保留原训练代码的首层/VAE/条件分支梯度检查；联合模式不要求初始与官方输出相等。

2026-10-21 实测迁移环境：SSH 用户 tanhe 的 UID/GID 为 1000，原模型/启动器要求 1005；没有 Docker；缺 `/mnt/hlk/workspace/Scheme_A&Scheme_B` 与 recipe 源码；清单包含 448 条序列、1344 条相机记录，但对应 source 路径全部不可访问，且全部没有 geometry_depth。官方权重虽在迁移的 harmonizer 目录中，原权重路径和依赖加载尚未适配。

因此本次交付是**已接入并通过独立测试的可选实现与配置**。没有启动整模型训练，没有计算质量指标，也没有宣称某种模式优于其他模式。原有第 1000 步检查点来自 1024×576 训练，不应被表述为新方案或 512×288 的训练成果。

## 参考与归属

- 联合 token 注意力思路参考 [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)。本文实现限制为同相机 S/M，保留独立跨相机路由，并非论文完整模型复现。
- 条件归一化思路参考 [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)。空间 M 调制是本项目扩展。
- 用户给出的 CVPR 2026 Correspondence-Attention PDF 链接本次访问返回 403，无法核实正文。本几何方案按明确的相机投影/深度一致性契约独立实现，不宣称复现该文方法。
