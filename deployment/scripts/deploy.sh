#!/bin/bash
# 模型部署脚本

set -e

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_PATH="${PROJECT_ROOT}/models/best_model.pth"
CONFIG_PATH="${PROJECT_ROOT}/configs/deployment_config.yaml"
LOG_FILE="${PROJECT_ROOT}/logs/deployment.log"

# 创建必要目录
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/output"

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 错误处理
error_exit() {
    log "错误: $1"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        error_exit "Python3 未安装"
    fi
    
    # 检查pip
    if ! command -v pip3 &> /dev/null; then
        error_exit "pip3 未安装"
    fi
    
    # 检查Docker (如果使用Docker部署)
    if [[ "$DEPLOYMENT_TYPE" == "docker" ]] && ! command -v docker &> /dev/null; then
        error_exit "Docker 未安装"
    fi
    
    log "依赖检查完成"
}

# 安装Python依赖
install_dependencies() {
    log "安装Python依赖..."
    
    if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
        pip3 install -r "${PROJECT_ROOT}/requirements.txt"
    else
        # 安装基本依赖
        pip3 install torch torchvision
        pip3 install flask fastapi uvicorn
        pip3 install numpy scikit-learn
        pip3 install psutil GPUtil
        pip3 install pyyaml
        log "已安装基本依赖"
    fi
    
    log "依赖安装完成"
}

# 系统分析
analyze_system() {
    log "分析系统配置..."
    
    # CPU核心数
    CPU_CORES=$(nproc)
    log "CPU核心数: $CPU_CORES"
    
    # 内存大小
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    log "内存大小: ${MEMORY_GB}GB"
    
    # GPU检查
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log "GPU数量: $GPU_COUNT"
    else
        GPU_COUNT=0
        log "未检测到GPU"
    fi
    
    # 存储空间
    STORAGE_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    log "可用存储空间: ${STORAGE_GB}GB"
}

# 选择部署类型
select_deployment_type() {
    log "选择部署类型..."
    
    if [[ -n "$DEPLOYMENT_TYPE" ]]; then
        log "使用指定部署类型: $DEPLOYMENT_TYPE"
        return
    fi
    
    # 基于系统配置自动选择
    if [[ $MEMORY_GB -ge 16 ]] && [[ $CPU_CORES -ge 8 ]] && [[ $GPU_COUNT -ge 1 ]]; then
        DEPLOYMENT_TYPE="kubernetes"
    elif [[ $MEMORY_GB -ge 8 ]] && [[ $CPU_CORES -ge 4 ]]; then
        DEPLOYMENT_TYPE="docker"
    else
        DEPLOYMENT_TYPE="local"
    fi
    
    log "自动选择部署类型: $DEPLOYMENT_TYPE"
}

# 本地部署
deploy_local() {
    log "开始本地部署..."
    
    # 启动Flask服务
    cd "$PROJECT_ROOT"
    
    # 设置环境变量
    export MODEL_PATH="$MODEL_PATH"
    export CONFIG_PATH="$CONFIG_PATH"
    export PORT="${PORT:-8080}"
    export WORKERS="${WORKERS:-4}"
    export BATCH_SIZE="${BATCH_SIZE:-32}"
    export MAX_MEMORY_MB="${MAX_MEMORY_MB:-2048}"
    export ENABLE_GPU="${ENABLE_GPU:-false}"
    export ENABLE_CACHING="${ENABLE_CACHING:-true}"
    export CACHE_SIZE_MB="${CACHE_SIZE_MB:-512}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    # 启动服务
    python3 -m flask run --host=0.0.0.0 --port=$PORT &
    SERVER_PID=$!
    
    log "本地服务已启动 (PID: $SERVER_PID)"
    log "服务地址: http://localhost:$PORT"
    
    # 保存PID
    echo $SERVER_PID > "${PROJECT_ROOT}/logs/server.pid"
    
    # 等待服务启动
    sleep 5
    
    # 健康检查
    if curl -f "http://localhost:$PORT/health" &> /dev/null; then
        log "健康检查通过"
    else
        error_exit "健康检查失败"
    fi
}

# Docker部署
deploy_docker() {
    log "开始Docker部署..."
    
    cd "$PROJECT_ROOT"
    
    # 构建Docker镜像
    log "构建Docker镜像..."
    docker build -t traffic-model:latest .
    
    # 运行容器
    log "启动Docker容器..."
    docker run -d \
        --name traffic-model-container \
        -p "${PORT:-8080}:8080" \
        -v "$(pwd)/models:/app/models" \
        -e MODEL_PATH="/app/models/best_model.pth" \
        -e WORKERS="${WORKERS:-4}" \
        -e BATCH_SIZE="${BATCH_SIZE:-32}" \
        -e MAX_MEMORY_MB="${MAX_MEMORY_MB:-2048}" \
        -e ENABLE_GPU="${ENABLE_GPU:-false}" \
        -e ENABLE_CACHING="${ENABLE_CACHING:-true}" \
        -e CACHE_SIZE_MB="${CACHE_SIZE_MB:-512}" \
        --memory="${MAX_MEMORY_MB}m" \
        --cpus="${WORKERS:-4}" \
        traffic-model:latest
    
    log "Docker容器已启动"
    log "服务地址: http://localhost:${PORT:-8080}"
    
    # 等待容器启动
    sleep 10
    
    # 检查容器状态
    if docker ps | grep -q traffic-model-container; then
        log "容器运行正常"
    else
        error_exit "容器启动失败"
    fi
}

# 优化模型
optimize_model() {
    log "开始模型优化..."
    
    cd "$PROJECT_ROOT"
    
    # 运行优化脚本
    python3 scripts/optimize_model.py \
        --config configs/optimization_config.yaml \
        --model_path "$MODEL_PATH" \
        --full_pipeline
    
    log "模型优化完成"
}

# 性能测试
performance_test() {
    log "开始性能测试..."
    
    PORT="${PORT:-8080}"
    
    # 简单性能测试
    for i in {1..10}; do
        start_time=$(date +%s.%N)
        curl -s "http://localhost:$PORT/health" > /dev/null
        end_time=$(date +%s.%N)
        
        response_time=$(echo "$end_time - $start_time" | bc)
        log "请求 $i 响应时间: ${response_time}s"
    done
    
    log "性能测试完成"
}

# 停止服务
stop_service() {
    log "停止服务..."
    
    if [[ "$DEPLOYMENT_TYPE" == "local" ]]; then
        if [[ -f "${PROJECT_ROOT}/logs/server.pid" ]]; then
            SERVER_PID=$(cat "${PROJECT_ROOT}/logs/server.pid")
            kill $SERVER_PID 2>/dev/null || true
            rm -f "${PROJECT_ROOT}/logs/server.pid"
            log "本地服务已停止"
        fi
    elif [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
        docker stop traffic-model-container 2>/dev/null || true
        docker rm traffic-model-container 2>/dev/null || true
        log "Docker容器已停止"
    fi
}

# 清理资源
cleanup() {
    log "清理资源..."
    
    # 清理缓存
    rm -rf "${PROJECT_ROOT}/model_cache" 2>/dev/null || true
    
    # 清理临时文件
    find "${PROJECT_ROOT}/output" -name "*.tmp" -delete 2>/dev/null || true
    
    log "资源清理完成"
}

# 显示帮助
show_help() {
    cat << EOF
模型部署脚本

用法: $0 [选项]

选项:
    --deploy-type TYPE     部署类型 (local, docker, kubernetes)
    --model-path PATH      模型文件路径
    --port PORT           服务端口 (默认: 8080)
    --workers NUM         工作进程数 (默认: 4)
    --batch-size NUM      批处理大小 (默认: 32)
    --enable-gpu          启用GPU支持
    --optimize            运行模型优化
    --test                运行性能测试
    --stop                停止服务
    --cleanup             清理资源
    --help               显示此帮助信息

环境变量:
    DEPLOYMENT_TYPE       部署类型
    MODEL_PATH           模型文件路径
    PORT                 服务端口
    WORKERS              工作进程数
    BATCH_SIZE           批处理大小
    ENABLE_GPU           启用GPU (true/false)
    ENABLE_CACHING       启用缓存 (true/false)
    CACHE_SIZE_MB        缓存大小(MB)
    MAX_MEMORY_MB        最大内存使用(MB)
    LOG_LEVEL            日志级别

示例:
    $0 --deploy-type local --optimize
    $0 --deploy-type docker --port 8080 --enable-gpu
    $0 --stop
    $0 --cleanup
EOF
}

# 主函数
main() {
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --deploy-type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --port)
                PORT="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --enable-gpu)
                ENABLE_GPU="true"
                shift
                ;;
            --optimize)
                OPTIMIZE=true
                shift
                ;;
            --test)
                TEST=true
                shift
                ;;
            --stop)
                stop_service
                exit 0
                ;;
            --cleanup)
                cleanup
                exit 0
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认值
    DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-local}"
    PORT="${PORT:-8080}"
    WORKERS="${WORKERS:-4}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    ENABLE_GPU="${ENABLE_GPU:-false}"
    ENABLE_CACHING="${ENABLE_CACHING:-true}"
    CACHE_SIZE_MB="${CACHE_SIZE_MB:-512}"
    MAX_MEMORY_MB="${MAX_MEMORY_MB:-2048}"
    LOG_LEVEL="${LOG_LEVEL:-INFO}"
    
    log "开始部署流程"
    log "部署类型: $DEPLOYMENT_TYPE"
    log "模型路径: $MODEL_PATH"
    log "端口: $PORT"
    
    # 执行部署步骤
    check_dependencies
    install_dependencies
    analyze_system
    select_deployment_type
    
    # 模型优化
    if [[ "$OPTIMIZE" == "true" ]]; then
        optimize_model
    fi
    
    # 执行部署
    case $DEPLOYMENT_TYPE in
        local)
            deploy_local
            ;;
        docker)
            deploy_docker
            ;;
        *)
            error_exit "不支持的部署类型: $DEPLOYMENT_TYPE"
            ;;
    esac
    
    # 性能测试
    if [[ "$TEST" == "true" ]]; then
        performance_test
    fi
    
    log "部署完成!"
    log "服务地址: http://localhost:$PORT"
    log "健康检查: http://localhost:$PORT/health"
    
    # 等待用户中断
    log "按 Ctrl+C 停止服务"
    trap 'log "接收到中断信号，正在停止服务..."; stop_service; exit 0' INT
    
    # 保持脚本运行
    while true; do
        sleep 1
    done
}

# 脚本入口点
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi