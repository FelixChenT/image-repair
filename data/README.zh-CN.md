# 数据集和训练指南

本文档介绍了如何收集数据、准备环境以及训练图像修复模型。本文档专为新手编写，因此每个步骤都将详细说明。

## 1. 硬件清单
- **最低可行配置**：8 核 CPU、16 GB 内存和至少 50 GB 的可用磁盘空间。训练将在 CPU 上运行，但速度会非常慢（对于中型数据集可能需要数天）。
- **推荐配置**：配备 8 GB VRAM 的 NVIDIA GPU（例如 RTX 3060 或更高版本）、32 GB 系统内存、SSD 存储。更强大的 GPU 可将训练时间从数天缩短至数小时。
- **可选附件**：用于存储原始扫描件的外部驱动器，以及用于检查视觉效果的校准显示器。

## 2. 软件先决条件
- Windows 10/11、macOS 12+ 或最新的 Linux 发行版。
- [Python 3.10+](https://www.python.org/downloads/) 并已安装 `pip`。
- 用于下载存储库的 [Git](https://git-scm.com/)（或手动下载 ZIP 存档）。
- 如果您计划在 NVIDIA GPU 上进行训练，请确保已安装最新的 GPU 驱动程序和 CUDA 工具包。

## 3. 项目设置（每台机器一次）
1. 克隆或解压缩存储库。
   ```bash
   git clone https://github.com/your-org/image-repair.git
   cd image-repair
   ```
2. 创建并激活虚拟环境。
   ```bash
   python -m venv .venv
   # PowerShell
   . .venv/Scripts/Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```
3. 安装依赖项。
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. （可选但推荐）启用格式化和 linting 钩子。
   ```bash
   pre-commit install
   ```

## 4. 整理您的数据集
1. 收集原始照片（PNG、JPG 或 TIFF）。分辨率更高（短边 > 1024 像素）的图像会产生更好的效果。
2. 将所有图像放入一个专用目录中，例如：
   ```
   data/
     raw_photos/
       family_001.jpg
       family_002.png
       ...
   ```
3. 保持简单的命名方案（无空格、ASCII 字符）以避免路径问题。
4. 确保每张图像都是 RGB 格式。对于黑白扫描件，请使用任何图像编辑器或以下脚本将其转换为 RGB：
   ```bash
   python - <<'PY'
   from pathlib import Path
   from PIL import Image

   root = Path('data/raw_photos')
   for path in root.rglob('*'):
       if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}:
           image = Image.open(path).convert('RGB')
           image.save(path)
   PY
   ```

## 5. 可选的预处理技巧
- 在训练前裁剪掉边框或扫描仪伪影。
- 如果您的扫描件尺寸差异很大，请考虑调整它们的大小，使最短边在 720 到 1080 像素之间。
- 如果您计划监控泛化能力，请将验证子集（例如，10% 的图像）存储在一个单独的文件夹中。

## 6. 开始训练
该存储库包含 `scripts/train.py`，它使用合成退化来训练缺陷修复模块和多尺度超分辨率模型。

### 基本训练运行
```bash
python scripts/train.py \
  --data-root data/raw_photos \
  --epochs 10 \
  --batch-size 4 \
  --patch-size 128 \
  --scale 4 \
  --device cuda \
  --output-dir outputs/checkpoints
```
- 如果您没有 GPU，请设置 `--device cpu`（预计运行时间会非常长）。
- `--patch-size` 必须能被 `--scale` 整除。如果您的 GPU 有足够的内存，请增加到 192 或 256。
- 检查点将写入 `outputs/checkpoints`，每个 epoch 一个文件。

### 推荐的监控
- 在控制台中观察 **SR Loss**、**Repair Loss**、**PSNR** 和 **SSIM**。PSNR/SSIM 的上升表明重建效果更好。
- 如果您需要更丰富的图表，可以通过扩展 `train.py` 来使用日志记录工具（例如 [Weights & Biases](https://wandb.ai/)、TensorBoard）。

### 恢复训练
要从检查点恢复，请传递 `--resume outputs/checkpoints/checkpoint_epoch_010.pth`（在扩展脚本后添加此标志；如果需要，可以轻松实现一个简单的恢复助手）。

## 7. 验证结果
1. 将少量未经处理的照片复制到 `assets/samples`（或您喜欢的任何文件夹）中。
2. 使用您训练好的模型运行推理（如果您将检查点存储在其他地方，请更新路径）：
   ```bash
   python scripts/run_inference.py \
     --input assets/samples \
     --output outputs/demo \
     --scale 4 \
     --device cuda \
     --save-intermediate
   ```
3. 检查生成的文件：
   - `_enhanced.png`：经过色彩和噪声调整后的最终结果。
   - `_repaired.png`、`_x4.png`、`_color.png`：有助于调试问题的中间输出。

## 8. 实用建议
- **从小处着手**：在十几张照片上运行 1-2 个 epoch，以验证一切正常，然后再启动长时间的训练。
- **备份**：将原始扫描件和检查点保存在安全的位置（云存储或外部驱动器）。
- **版本控制**：提交配置更改（学习率、补丁大小），以便以后可以重现结果。
- **质量检查**：将 PSNR/SSIM 与验证子集进行比较；大幅下降通常表示过拟合或数据问题。

## 9. 故障排除
- `CUDA out of memory`：降低 `--batch-size`、减小 `--patch-size` 或切换到 `--device cpu`。
- `No image files discovered`：仔细检查 `--data-root` 路径和文件扩展名。
- `Training is extremely slow`：确认脚本正在 GPU 上运行（`nvidia-smi` 应显示活动）或升级硬件。
- `Results look blurry`：继续训练更多 epoch，收集更高分辨率的数据，或尝试更大的补丁大小。

通过这些步骤，即使是新手也可以从原始照片到训练好的修复模型。继续优化数据集和超参数，以达到生产就绪的质量。