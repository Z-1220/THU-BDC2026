# 部署到远程服务器

将当前分支的代码推送到 Gitee，并在远程服务器上拉取更新并运行。

## Step 1: 检查工作区
<execute_command>
<command>git status --porcelain</command>
</execute_command>
如有未提交变更，询问用户 commit message。

## Step 2: 提交并推送
<execute_command>
<command>git add -A && git commit -m "{用户输入}" && git push origin $(git branch --show-current)</command>
<requires_approval>true</requires_approval>
</execute_command>

## Step 3: 远程部署
连接到服务器，拉取最新代码，使用 uv 同步依赖，并在 tmux 中启动服务。

<execute_command>
<command>ssh -o StrictHostKeyChecking=no ubuntu@10.155.12.21 "cd /home/ubuntu/competition/Z2025925435/THU-BDC2026 && git pull && uv sync && tmux kill-session -t myapp 2>/dev/null; tmux new-session -d -s myapp 'uv run python code/src/train.py'"</command>
<requires_approval>true</requires_approval>
</execute_command>

## Step 4: 验证部署
<execute_command>
<command>ssh ubuntu@10.155.12.21 "tmux ls"</command>
</execute_command>
检查 `myapp` 会话是否存在。