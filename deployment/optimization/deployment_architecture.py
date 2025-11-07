"""
部署架构优化模块
支持普通笔记本和云服务器部署
"""

import os
import json
import yaml
import docker
import subprocess
import psutil
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import requests
import socket
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentType(Enum):
    """部署类型"""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    CLOUD = "cloud"


class ResourceType(Enum):
    """资源类型"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    STORAGE = "storage"


@dataclass
class SystemSpecs:
    """系统规格"""
    cpu_cores: int
    memory_gb: float
    gpu_count: int
    gpu_memory_gb: float
    storage_gb: float
    os_type: str
    architecture: str


@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_type: DeploymentType
    model_name: str
    model_version: str
    port: int
    workers: int
    batch_size: int
    max_memory_mb: int
    enable_gpu: bool
    enable_caching: bool
    cache_size_mb: int
    log_level: str
    environment: str


class SystemAnalyzer:
    """系统分析器"""
    
    @staticmethod
    def get_system_specs() -> SystemSpecs:
        """获取系统规格"""
        # CPU信息
        cpu_cores = psutil.cpu_count(logical=True)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        # GPU信息
        gpu_count = 0
        gpu_memory_gb = 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            gpu_count = len(gpus)
            gpu_memory_gb = sum([gpu.memoryTotal for gpu in gpus]) / 1024
        except ImportError:
            logger.warning("GPUtil未安装，无法检测GPU信息")
        
        # 存储信息
        storage = psutil.disk_usage('/')
        storage_gb = storage.total / (1024**3)
        
        # 操作系统信息
        import platform
        os_type = platform.system()
        architecture = platform.machine()
        
        return SystemSpecs(
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            storage_gb=storage_gb,
            os_type=os_type,
            architecture=architecture
        )
    
    @staticmethod
    def analyze_deployment_suitability(specs: SystemSpecs) -> Dict[str, Any]:
        """分析部署适合性"""
        analysis = {
            'recommended_deployment': None,
            'resource_rating': {},
            'optimization_suggestions': [],
            'limitations': []
        }
        
        # 评估CPU
        cpu_rating = min(specs.cpu_cores / 4, 1.0)  # 4核为满分
        analysis['resource_rating']['cpu'] = cpu_rating
        
        # 评估内存
        memory_rating = min(specs.memory_gb / 8, 1.0)  # 8GB为满分
        analysis['resource_rating']['memory'] = memory_rating
        
        # 评估GPU
        gpu_rating = 0.0
        if specs.gpu_count > 0:
            gpu_rating = min(specs.gpu_memory_gb / 4, 1.0)  # 4GB为满分
        analysis['resource_rating']['gpu'] = gpu_rating
        
        # 评估存储
        storage_rating = min(specs.storage_gb / 100, 1.0)  # 100GB为满分
        analysis['resource_rating']['storage'] = storage_rating
        
        # 推荐部署类型
        if specs.memory_gb >= 16 and specs.cpu_cores >= 8 and specs.gpu_count >= 1:
            analysis['recommended_deployment'] = DeploymentType.KUBERNETES
        elif specs.memory_gb >= 8 and specs.cpu_cores >= 4:
            analysis['recommended_deployment'] = DeploymentType.DOCKER
        else:
            analysis['recommended_deployment'] = DeploymentType.LOCAL
        
        # 优化建议
        if specs.memory_gb < 4:
            analysis['optimization_suggestions'].append("建议增加内存以提高性能")
        
        if specs.cpu_cores < 2:
            analysis['optimization_suggestions'].append("建议使用更小的模型或减少并发数")
        
        if specs.gpu_count == 0:
            analysis['optimization_suggestions'].append("建议使用CPU优化版本的模型")
        
        if gpu_rating < 0.5:
            analysis['limitations'].append("GPU内存不足，可能需要使用较小的模型")
        
        return analysis


class LocalDeploymentManager:
    """本地部署管理器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.process = None
        self.health_check_url = f"http://localhost:{config.port}/health"
        
    def deploy(self, model_path: str, script_path: str) -> bool:
        """部署模型"""
        try:
            # 检查端口是否被占用
            if self._is_port_in_use(self.config.port):
                logger.error(f"端口 {self.config.port} 已被占用")
                return False
            
            # 启动服务
            cmd = [
                'python', script_path,
                '--model_path', model_path,
                '--port', str(self.config.port),
                '--workers', str(self.config.workers),
                '--batch_size', str(self.config.batch_size),
                '--max_memory_mb', str(self.config.max_memory_mb),
                '--enable_gpu', str(self.config.enable_gpu),
                '--enable_caching', str(self.config.enable_caching),
                '--cache_size_mb', str(self.config.cache_size_mb),
                '--log_level', self.config.log_level
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服务启动
            time.sleep(5)
            
            # 健康检查
            if self._health_check():
                logger.info(f"本地部署成功，服务运行在端口 {self.config.port}")
                return True
            else:
                logger.error("健康检查失败")
                return False
                
        except Exception as e:
            logger.error(f"本地部署失败: {e}")
            return False
    
    def _is_port_in_use(self, port: int) -> bool:
        """检查端口是否被占用"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _health_check(self) -> bool:
        """健康检查"""
        try:
            response = requests.get(self.health_check_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop(self):
        """停止服务"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            logger.info("本地服务已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        if not self.process:
            return {'status': 'not_running'}
        
        return {
            'status': 'running' if self.process.poll() is None else 'stopped',
            'pid': self.process.pid,
            'port': self.config.port
        }


class DockerDeploymentManager:
    """Docker部署管理器"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = docker.from_env()
        self.container = None
        
    def create_dockerfile(self, base_image: str = "python:3.9-slim") -> str:
        """创建Dockerfile"""
        dockerfile_content = f"""
FROM {base_image}

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_CACHE_SIZE={self.config.cache_size_mb}
ENV MAX_MEMORY_MB={self.config.max_memory_mb}

# 暴露端口
EXPOSE {self.config.port}

# 启动命令
CMD ["python", "server.py", "--port", "{self.config.port}", "--workers", "{self.config.workers}"]
"""
        return dockerfile_content
    
    def build_image(self, dockerfile_path: str, image_name: str) -> bool:
        """构建Docker镜像"""
        try:
            # 构建镜像
            image = self.client.images.build(
                path=os.path.dirname(dockerfile_path),
                dockerfile=dockerfile_path,
                tag=image_name
            )
            
            logger.info(f"Docker镜像构建成功: {image_name}")
            return True
            
        except Exception as e:
            logger.error(f"Docker镜像构建失败: {e}")
            return False
    
    def deploy(self, image_name: str, model_path: str) -> bool:
        """部署服务"""
        try:
            # 准备挂载路径
            model_volume = f"{os.path.abspath(model_path)}:/app/model"
            
            # 运行容器
            self.container = self.client.containers.run(
                image_name,
                detach=True,
                ports={f'{self.config.port}/tcp': self.config.port},
                volumes=[model_volume],
                environment={
                    'MODEL_PATH': '/app/model',
                    'WORKERS': str(self.config.workers),
                    'BATCH_SIZE': str(self.config.batch_size),
                    'MAX_MEMORY_MB': str(self.config.max_memory_mb),
                    'ENABLE_GPU': str(self.config.enable_gpu),
                    'ENABLE_CACHING': str(self.config.enable_caching),
                    'CACHE_SIZE_MB': str(self.config.cache_size_mb)
                },
                mem_limit=f"{self.config.max_memory_mb}m",
                cpus=self.config.workers
            )
            
            # 等待容器启动
            time.sleep(10)
            
            # 检查容器状态
            container_info = self.client.containers.get(self.container.id)
            if container_info.status == 'running':
                logger.info(f"Docker部署成功，容器ID: {self.container.id}")
                return True
            else:
                logger.error("容器启动失败")
                return False
                
        except Exception as e:
            logger.error(f"Docker部署失败: {e}")
            return False
    
    def stop(self):
        """停止服务"""
        if self.container:
            self.container.stop()
            self.container.remove()
            logger.info("Docker容器已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        if not self.container:
            return {'status': 'not_running'}
        
        try:
            container_info = self.client.containers.get(self.container.id)
            return {
                'status': container_info.status,
                'id': self.container.id,
                'port': self.config.port,
                'image': container_info.image.tags[0] if container_info.image.tags else 'unknown'
            }
        except:
            return {'status': 'error'}


class CloudDeploymentManager:
    """云部署管理器"""
    
    def __init__(self, config: DeploymentConfig, cloud_config: Dict):
        self.config = config
        self.cloud_config = cloud_config
        self.deployment_id = None
        self.service_url = None
        
    def deploy(self, model_path: str) -> bool:
        """云端部署"""
        try:
            # 这里实现具体的云平台部署逻辑
            # 例如：AWS Lambda, Azure Functions, Google Cloud Run等
            
            platform = self.cloud_config.get('platform', 'aws')
            
            if platform == 'aws':
                return self._deploy_aws(model_path)
            elif platform == 'azure':
                return self._deploy_azure(model_path)
            elif platform == 'gcp':
                return self._deploy_gcp(model_path)
            else:
                logger.error(f"不支持的云平台: {platform}")
                return False
                
        except Exception as e:
            logger.error(f"云部署失败: {e}")
            return False
    
    def _deploy_aws(self, model_path: str) -> bool:
        """AWS部署"""
        # 实现AWS部署逻辑
        # 这里只是一个示例
        logger.info("AWS部署功能需要根据具体需求实现")
        return True
    
    def _deploy_azure(self, model_path: str) -> bool:
        """Azure部署"""
        # 实现Azure部署逻辑
        logger.info("Azure部署功能需要根据具体需求实现")
        return True
    
    def _deploy_gcp(self, model_path: str) -> bool:
        """GCP部署"""
        # 实现GCP部署逻辑
        logger.info("GCP部署功能需要根据具体需求实现")
        return True
    
    def stop(self):
        """停止云服务"""
        if self.deployment_id:
            # 实现停止云服务的逻辑
            logger.info(f"云服务 {self.deployment_id} 已停止")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'status': 'deployed' if self.deployment_id else 'not_deployed',
            'deployment_id': self.deployment_id,
            'service_url': self.service_url
        }


class DeploymentOrchestrator:
    """部署编排器"""
    
    def __init__(self):
        self.deployment_managers = {}
        self.current_deployment = None
        
    def analyze_and_recommend(self, specs: SystemSpecs) -> Dict[str, Any]:
        """分析并推荐部署方案"""
        analyzer = SystemAnalyzer()
        analysis = analyzer.analyze_deployment_suitability(specs)
        
        # 创建推荐配置
        recommended_type = analysis['recommended_deployment']
        
        if recommended_type == DeploymentType.LOCAL:
            config = self._create_local_config(specs)
        elif recommended_type == DeploymentType.DOCKER:
            config = self._create_docker_config(specs)
        elif recommended_type == DeploymentType.KUBERNETES:
            config = self._create_k8s_config(specs)
        else:
            config = self._create_local_config(specs)
        
        return {
            'system_analysis': analysis,
            'recommended_config': asdict(config),
            'deployment_instructions': self._get_deployment_instructions(recommended_type)
        }
    
    def _create_local_config(self, specs: SystemSpecs) -> DeploymentConfig:
        """创建本地部署配置"""
        workers = min(specs.cpu_cores, 4)
        max_memory_mb = int(specs.memory_gb * 1024 * 0.8)  # 使用80%内存
        
        return DeploymentConfig(
            deployment_type=DeploymentType.LOCAL,
            model_name="traffic_prediction_model",
            model_version="1.0",
            port=8080,
            workers=workers,
            batch_size=32,
            max_memory_mb=max_memory_mb,
            enable_gpu=specs.gpu_count > 0,
            enable_caching=True,
            cache_size_mb=min(512, max_memory_mb // 4),
            log_level="INFO",
            environment="production"
        )
    
    def _create_docker_config(self, specs: SystemSpecs) -> DeploymentConfig:
        """创建Docker部署配置"""
        workers = min(specs.cpu_cores, 8)
        max_memory_mb = int(specs.memory_gb * 1024 * 0.6)  # Docker使用60%内存
        
        return DeploymentConfig(
            deployment_type=DeploymentType.DOCKER,
            model_name="traffic_prediction_model",
            model_version="1.0",
            port=8080,
            workers=workers,
            batch_size=64,
            max_memory_mb=max_memory_mb,
            enable_gpu=specs.gpu_count > 0,
            enable_caching=True,
            cache_size_mb=min(1024, max_memory_mb // 2),
            log_level="INFO",
            environment="production"
        )
    
    def _create_k8s_config(self, specs: SystemSpecs) -> DeploymentConfig:
        """创建Kubernetes部署配置"""
        workers = min(specs.cpu_cores, 16)
        max_memory_mb = int(specs.memory_gb * 1024 * 0.7)
        
        return DeploymentConfig(
            deployment_type=DeploymentType.KUBERNETES,
            model_name="traffic_prediction_model",
            model_version="1.0",
            port=8080,
            workers=workers,
            batch_size=128,
            max_memory_mb=max_memory_mb,
            enable_gpu=specs.gpu_count > 0,
            enable_caching=True,
            cache_size_mb=min(2048, max_memory_mb // 2),
            log_level="INFO",
            environment="production"
        )
    
    def _get_deployment_instructions(self, deployment_type: DeploymentType) -> List[str]:
        """获取部署说明"""
        instructions = {
            DeploymentType.LOCAL: [
                "1. 确保Python 3.8+已安装",
                "2. 安装依赖: pip install -r requirements.txt",
                "3. 运行服务: python server.py",
                "4. 访问: http://localhost:8080/health"
            ],
            DeploymentType.DOCKER: [
                "1. 安装Docker和Docker Compose",
                "2. 构建镜像: docker build -t traffic-model .",
                "3. 运行容器: docker run -p 8080:8080 traffic-model",
                "4. 访问: http://localhost:8080/health"
            ],
            DeploymentType.KUBERNETES: [
                "1. 确保kubectl和k8s集群可用",
                "2. 应用配置: kubectl apply -f k8s-deployment.yaml",
                "3. 检查状态: kubectl get pods",
                "4. 暴露服务: kubectl port-forward svc/traffic-model 8080:8080"
            ]
        }
        
        return instructions.get(deployment_type, [])
    
    def deploy(self, config: DeploymentConfig, model_path: str, 
               script_path: str = None, cloud_config: Dict = None) -> bool:
        """执行部署"""
        try:
            if config.deployment_type == DeploymentType.LOCAL:
                manager = LocalDeploymentManager(config)
                success = manager.deploy(model_path, script_path or "server.py")
                
            elif config.deployment_type == DeploymentType.DOCKER:
                manager = DockerDeploymentManager(config)
                success = manager.deploy("traffic-model:latest", model_path)
                
            elif config.deployment_type == DeploymentType.CLOUD:
                manager = CloudDeploymentManager(config, cloud_config or {})
                success = manager.deploy(model_path)
                
            else:
                logger.error(f"不支持的部署类型: {config.deployment_type}")
                return False
            
            if success:
                self.current_deployment = manager
                logger.info(f"部署成功: {config.deployment_type.value}")
                return True
            else:
                logger.error("部署失败")
                return False
                
        except Exception as e:
            logger.error(f"部署异常: {e}")
            return False
    
    def stop_current_deployment(self):
        """停止当前部署"""
        if self.current_deployment:
            self.current_deployment.stop()
            self.current_deployment = None
            logger.info("当前部署已停止")
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        if not self.current_deployment:
            return {'status': 'no_active_deployment'}
        
        return self.current_deployment.get_status()


def save_deployment_config(config: DeploymentConfig, file_path: str):
    """保存部署配置"""
    with open(file_path, 'w') as f:
        yaml.dump(asdict(config), f, default_flow_style=False)


def load_deployment_config(file_path: str) -> DeploymentConfig:
    """加载部署配置"""
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return DeploymentConfig(**data)


if __name__ == "__main__":
    # 示例使用
    print("=== 系统分析测试 ===")
    
    # 获取系统规格
    specs = SystemAnalyzer.get_system_specs()
    print(f"系统规格: CPU {specs.cpu_cores}核, 内存 {specs.memory_gb:.1f}GB, "
          f"GPU {specs.gpu_count}个, 存储 {specs.storage_gb:.1f}GB")
    
    # 创建部署编排器
    orchestrator = DeploymentOrchestrator()
    
    # 分析并推荐部署方案
    recommendation = orchestrator.analyze_and_recommend(specs)
    print(f"推荐部署类型: {recommendation['system_analysis']['recommended_deployment']}")
    print(f"优化建议: {recommendation['system_analysis']['optimization_suggestions']}")
    
    # 保存推荐配置
    config = DeploymentConfig(**recommendation['recommended_config'])
    save_deployment_config(config, "deployment_config.yaml")
    
    print("部署架构分析完成")