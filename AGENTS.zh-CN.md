# 仓库指南

## 项目结构和模块组织
将生产代码保存在 `src/` 目录下，并分为以下几个部分：`src/data/` 用于数据加载器和增强功能，`src/models/` 用于超分辨率和修复网络，`src/pipelines/` 用于端到端编排。轻量级实用程序（如指标、日志记录）应放在 `src/utils/` 中。命令行入口点和实验脚本位于 `scripts/` 目录中（例如 `scripts/run_inference.py`）。测试代码的目录结构应与 `src/` 保持一致，位于 `tests/` 目录下。Jupyter 实验笔记本应保存在 `notebooks/` 中，并清除输出后提交。将小型演示资源存储在 `assets/` 中，并附上一个 `README` 文件说明其来源；大型数据集不应存放在仓库中，而是通过 `data/README.md` 进行引用。

## 构建、测试和开发命令
首先使用 `python -m venv .venv` 创建虚拟环境，然后运行 `./.venv/Scripts/activate` 激活环境，并执行 `pip install -r requirements.txt` 安装依赖。使用 `pre-commit install` 启用 linting 钩子。在每次推送前运行 `pre-commit run --all-files`。通过 `pytest`（单元测试）和 `pytest tests/integration`（完整流水线检查）执行主测试套件。使用 `python scripts/run_inference.py --input assets/samples --output outputs/demo` 快速预览模型效果。

## 编码风格和命名约定
遵循 PEP 8 规范，使用 4 个空格缩进，行长不超过 100 个字符。模块和函数名使用 `snake_case`，类名使用 `PascalCase`，实验配置文件使用 `kebab-case`（例如 `configs/sr-x4.yaml`）。使用 `black` 格式化代码，使用 `ruff` 进行 linting；在公共函数上强制使用类型提示，并优先使用 `numpy.typing` 或 `torch` 的相关注解。

## 测试指南
为每个模块编写单元测试（例如 `tests/models/test_srnet.py`）。优先使用基于缓存样本图像的确定性测试固件。对于恢复质量，应包含感知度量断言（PSNR、SSIM 阈值）。使用 `scripts/build_golden.py` 更新或重新生成黄金输出，并在 `tests/README.md` 中记录更改。目标是通过 `pytest --cov=src` 实现每个模块 85% 以上的测试覆盖率。

## 提交和拉取请求指南
目前尚无 git 历史记录；请从“Conventional Commits”开始（例如，`feat: add x4 super-resolution pipeline`）。每个拉取请求限制为一个功能或修复，描述所涉及的数据集和配置，并附上视觉变化的“之前/之后”缩略图。在可用时链接问题 ID，并注明任何大于 20 MB 且未纳入 git 的文件。

## 数据和安全说明
尊重隐私：未经匿名化处理或同意，请勿提交个人照片。将 API 密钥或模型权重存储在 `.env` 文件中，并通过 `.env.example` 进行跟踪，并每季度轮换共享凭据。