---
# 可选：添加条件路径，让规则仅在处理 Python 项目时激活
paths:
  - "**/*.py"
  - "**/pyproject.toml"
  - "**/requirements*.txt"
---

# Python 环境管理强制规范：必须使用 uv

本规则强制约束 Cline 在所有 Python 环境管理相关操作中**只能使用 `uv`**，严禁使用 `pip`、`pip3`、`conda`、`poetry`、`easy_install` 等替代工具。

## 核心原则
- **唯一允许的 Python 包管理工具：`uv`**
- 所有依赖安装、虚拟环境创建、脚本运行、包列表查看等操作都必须通过 `uv` 命令完成。
- 禁止生成任何包含 `pip install`、`python -m pip`、`conda install`、`poetry add` 的命令。

## 具体指令

### 1. 安装依赖
- ✅ 正确：`uv add <package>`
- ❌ 错误：`pip install -r requirements.txt`、`pip install <package>`、`conda install <package>`
- ❌ 错误：直接编辑 `requirements.txt` 后执行 `pip install`

### 2. 运行 Python 脚本/模块
- ✅ 正确：`uv run python script.py` 或 `uv run python -m module_name`
- ❌ 错误：`python script.py`、`./script.py`（除非已确保在 uv 管理的虚拟环境中）

### 3. 查看已安装包
- ✅ 正确：`uv pip list`、`uv pip show <package>`
- ❌ 错误：`pip list`、`pip freeze`

### 4. 创建虚拟环境
- ✅ 正确：`uv sync`或者`uv pin python python-version`后会自动创建环境
  > 如`uv python pin 3.10`
- ❌ 错误：`python -m venv venv`、`virtualenv venv`

### 5. 升级 uv 自身
- ✅ 正确：`uv self update`
- ❌ 错误：`pip install --upgrade uv`