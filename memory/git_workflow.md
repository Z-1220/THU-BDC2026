---
name: git-workflow
description: 个人开发者的 Git 分支与合入规范 — 新功能必须在独立分支开发，经手动验证后才能合入 main
metadata:
  type: project
---

所有代码变更必须遵循 Git 分支工作流，严禁直接在 main 上提交。

**分支约定**：
- `main` — 主开发分支，只接受已验证的功能分支合入
- `baseline` — 赛事方原始代码备份分支（历史快照，不在此之上开发）
- `feature/<name>` — 新功能分支，从 `main` 分出，开发完成后合回 `main`

**Why:** main 分支需要保持可运行状态，未经人工验证的代码不能混入。baseline 保留主办方原始代码作为参考基线。

**How to apply:**
1. 任何新功能开发，必须先 `git checkout -b feature/<descriptive-name>` 从 main 创建新分支
2. 开发完成后，由用户手动运行检查验证通过后，才允许合入 main
3. 不在 main 上直接提交、不 amend main 上的 commit、不对 main 执行 force push
4. 遵循合理的个人开发者 Git 习惯：有意义的 commit message、不提交无关文件、不跳过 hooks

**Commit message 规范**：
- 使用中文撰写，以动词开头说明变更内容。常见标记前缀可用英文：`fix:`、`feat:`、`refactor:`、`docs:`、`chore:`
- 例如：`feat: 添加行业中性化处理器`、`fix: 修复周五样本过滤缺失问题`
- 一条 commit 对应一个逻辑变更

**文件管理**：
- 及时更新 `.gitignore`，排除缓存目录（`__pycache__/`、`temp/qlib_data/`、`temp/mlruns/`）、环境文件（`.venv/`、`.env`）、IDE 配置、压缩包等
- 不要提交过大的数据文件（CSV、模型权重）到 git，这些应通过脚本生成或存放在外部
- `git add` 时使用具体路径，避免 `git add -A` 误提交无关文件
