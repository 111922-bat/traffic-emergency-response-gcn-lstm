#!/bin/bash

# 智能交通流预测系统启动脚本

echo "=== 智能交通流预测与应急管理系统 ==="
echo "正在启动生产级系统..."

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
echo "Python版本: $python_version"

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装Python依赖包..."
pip install -r requirements.txt

# 复制模型文件
echo "准备模型文件..."
if [ -d "/workspace/code/models" ]; then
    cp -r /workspace/code/models ./models
    echo "模型文件已复制"
else
    echo "警告: 未找到模型文件目录，将使用模拟模式"
fi

if [ -d "/workspace/code/services" ]; then
    cp -r /workspace/code/services ./services
    echo "服务文件已复制"
else
    echo "警告: 未找到服务文件目录，将使用模拟模式"
fi

if [ -d "/workspace/code/pathfinding" ]; then
    cp -r /workspace/code/pathfinding ./pathfinding
    echo "路径规划文件已复制"
else
    echo "警告: 未找到路径规划文件目录，将使用模拟模式"
fi

# 启动API服务器
echo "启动API服务器..."
echo "服务器将在 http://localhost:3001 启动"
echo "WebSocket将在 ws://localhost:3001 启动"
echo "按 Ctrl+C 停止服务器"
echo ""

python3 api_server.py