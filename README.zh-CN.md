# 图像修复工具包

一个通过自动修复缺陷和多尺度超分辨率来恢复老旧照片的参考项目。它捆绑了数据加载、缺陷模拟、色彩增强、去噪和质量评估等实用功能。

## 主要功能
- **多尺度超分辨率**: `MultiScaleSRNet` 可生成 x2/x3/x4 的输出，以匹配不同的分辨率目标。
- **缺陷检测和修复**: `DefectRepairModule` 结合了分割和修复网络，以处理划痕、污渍和折痕。
- **色彩恢复和去噪**: 轻量级的灰度世界平衡和高斯平滑实用程序可提高色彩保真度并抑制噪声。
- **质量评估**: 内置的 PSNR 和 SSIM 指标有助于验证修复质量。

## 项目布局
```
src/
  data/                # 数据集和缺陷模拟实用程序
  models/              # 超分辨率和缺陷修复网络
  pipelines/           # 端到端推理管道
  utils/               # 图像 I/O、指标、日志记录和色彩辅助工具
scripts/
  run_inference.py     # 命令行推理入口点
assets/                # 小型演示图像和文档
assets/README.md       # 示例资源的来源说明
tests/                 # 与源代码树镜像的单元测试
```

## 环境设置
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
通过 `pre-commit install` 安装一次 pre-commit 钩子，以保持格式一致。

## 运行测试
```bash
pytest
```

## 推理示例
```bash
python scripts/run_inference.py --input assets/samples --output outputs/demo --scale 4
```
添加 `--save-intermediate` 以保留中间修复阶段。

## 数据说明
大型训练数据集未检入存储库。在 `data/README.md` 中记录获取步骤或预处理脚本。