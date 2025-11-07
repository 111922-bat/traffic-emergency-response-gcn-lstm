#!/usr/bin/env python3
"""
依赖注入容器
Dependency Injection Container for Traffic Prediction System

提供松耦合的模块依赖管理，支持配置驱动和动态依赖解析
"""

import os
import sys
import logging
from typing import Dict, Type, TypeVar, Generic, Callable, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import importlib

# 类型定义
T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """模块配置"""
    name: str
    class_path: str
    factory: Optional[Callable] = None
    dependencies: list = None
    singleton: bool = True
    lazy_load: bool = True
    config: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


class ServiceLocator(Generic[T]):
    """服务定位器"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._configs: Dict[str, ModuleConfig] = {}
        self._singletons: Dict[str, Any] = {}
    
    def register(self, config: ModuleConfig):
        """注册服务配置"""
        self._configs[config.name] = config
        logger.info(f"注册服务: {config.name}")
    
    def get(self, name: str, *args, **kwargs) -> T:
        """获取服务实例"""
        if name not in self._configs:
            raise ValueError(f"未注册的服务: {name}")
        
        config = self._configs[name]
        
        # 单例模式
        if config.singleton and name in self._singletons:
            return self._singletons[name]
        
        # 懒加载
        if config.lazy_load:
            instance = self._create_instance(config, *args, **kwargs)
        else:
            instance = config.factory(*args, **kwargs) if config.factory else None
        
        if config.singleton:
            self._singletons[name] = instance
        
        return instance
    
    def _create_instance(self, config: ModuleConfig, *args, **kwargs) -> Any:
        """创建实例"""
        try:
            # 使用工厂函数
            if config.factory:
                return config.factory(*args, **kwargs)
            
            # 动态导入类
            module_path, class_name = config.class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            # 注入依赖
            if config.dependencies:
                dep_instances = [self.get(dep) for dep in config.dependencies]
                return cls(*dep_instances, *args, **kwargs)
            
            # 使用配置创建实例
            if config.config:
                instance = cls(**config.config)
                # 如果有额外的参数，使用setattr设置
                for key, value in kwargs.items():
                    setattr(instance, key, value)
                return instance
            
            return cls(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"创建实例失败 {config.name}: {e}")
            raise
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有服务实例"""
        return {name: self.get(name) for name in self._configs.keys()}


class PathResolver:
    """路径解析器"""
    
    def __init__(self, base_path: Optional[str] = None):
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._path_mappings: Dict[str, Path] = {}
    
    def add_mapping(self, alias: str, path: str):
        """添加路径映射"""
        resolved_path = self._resolve_path(path)
        self._path_mappings[alias] = resolved_path
        logger.debug(f"添加路径映射: {alias} -> {resolved_path}")
    
    def resolve(self, path_or_alias: str) -> Path:
        """解析路径"""
        # 如果是别名
        if path_or_alias in self._path_mappings:
            return self._path_mappings[path_or_alias]
        
        # 如果是相对路径
        if not os.path.isabs(path_or_alias):
            return self._base_path / path_or_alias
        
        return Path(path_or_alias)
    
    def _resolve_path(self, path: str) -> Path:
        """解析路径字符串"""
        if os.path.isabs(path):
            return Path(path)
        
        # 支持环境变量
        if path.startswith('$'):
            env_var = path[1:]
            env_value = os.getenv(env_var)
            if env_value:
                return Path(env_value)
        
        return self._base_path / path
    
    def get_project_root(self) -> Path:
        """获取项目根目录"""
        return self._base_path
    
    def get_module_path(self, module_name: str) -> Path:
        """获取模块路径"""
        return self.resolve(f"modules/{module_name}")
    
    def get_config_path(self, config_name: str) -> Path:
        """获取配置文件路径"""
        return self.resolve(f"config/{config_name}")
    
    def get_data_path(self, data_name: str) -> Path:
        """获取数据路径"""
        return self.resolve(f"data/{data_name}")


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, path_resolver: PathResolver):
        self.path_resolver = path_resolver
        self._configs: Dict[str, Any] = {}
        self._config_files: Dict[str, str] = {}
    
    def load_config(self, name: str, config_path: str, config_type: str = "yaml"):
        """加载配置文件"""
        resolved_path = self.path_resolver.resolve(config_path)
        
        if not resolved_path.exists():
            logger.warning(f"配置文件不存在: {resolved_path}")
            return
        
        try:
            if config_type.lower() == "yaml":
                import yaml
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_type.lower() == "json":
                import json
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"不支持的配置类型: {config_type}")
            
            self._configs[name] = config
            self._config_files[name] = str(resolved_path)
            logger.info(f"加载配置: {name} from {resolved_path}")
            
        except Exception as e:
            logger.error(f"加载配置失败 {name}: {e}")
            raise
    
    def get_config(self, name: str, default: Any = None) -> Any:
        """获取配置"""
        return self._configs.get(name, default)
    
    def set_config(self, name: str, config: Any):
        """设置配置"""
        self._configs[name] = config
        logger.info(f"设置配置: {name}")
    
    def get_config_value(self, name: str, key_path: str, default: Any = None) -> Any:
        """获取配置值（支持点号路径）"""
        config = self._configs.get(name)
        if not config:
            return default
        
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """更新配置"""
        if name not in self._configs:
            self._configs[name] = {}
        
        self._configs[name].update(updates)
        logger.info(f"更新配置: {name}")


class ModuleInterface(ABC):
    """模块接口"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]):
        """初始化模块"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """关闭模块"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取模块状态"""
        pass


class DependencyInjector:
    """依赖注入器"""
    
    def __init__(self, service_locator: ServiceLocator, config_manager: ConfigManager):
        self.service_locator = service_locator
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def inject_dependencies(self, instance: Any, config_prefix: str = ""):
        """注入依赖"""
        # 获取实例的配置
        if config_prefix:
            config = self.config_manager.get_config(config_prefix, {})
        else:
            config = {}
        
        # 注入配置属性
        for attr_name, attr_value in config.items():
            if hasattr(instance, attr_name):
                setattr(instance, attr_name, attr_value)
                self.logger.debug(f"注入配置属性: {attr_name}")
        
        # 注入依赖服务
        if hasattr(instance, '_dependencies'):
            for dep_name in instance._dependencies:
                try:
                    dep_instance = self.service_locator.get(dep_name)
                    setattr(instance, dep_name, dep_instance)
                    self.logger.debug(f"注入依赖服务: {dep_name}")
                except Exception as e:
                    self.logger.warning(f"注入依赖失败 {dep_name}: {e}")
    
    def create_module(self, module_config: ModuleConfig) -> ModuleInterface:
        """创建模块实例"""
        try:
            instance = self.service_locator.get(module_config.name)
            self.inject_dependencies(instance, module_config.name)
            
            if isinstance(instance, ModuleInterface):
                instance.initialize(module_config.config)
                return instance
            else:
                self.logger.warning(f"模块 {module_config.name} 未实现 ModuleInterface")
                return instance
                
        except Exception as e:
            self.logger.error(f"创建模块失败 {module_config.name}: {e}")
            raise


class ErrorHandlingStrategy(ABC):
    """错误处理策略"""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: str) -> Any:
        """处理错误"""
        pass


class FallbackErrorStrategy(ErrorHandlingStrategy):
    """回退错误策略"""
    
    def __init__(self, fallback_value: Any = None):
        self.fallback_value = fallback_value
    
    def handle_error(self, error: Exception, context: str) -> Any:
        """返回回退值"""
        logger.warning(f"错误处理 [{context}]: {error}, 使用回退值")
        return self.fallback_value


class RetryErrorStrategy(ErrorHandlingStrategy):
    """重试错误策略"""
    
    def __init__(self, max_retries: int = 3, delay: float = 1.0):
        self.max_retries = max_retries
        self.delay = delay
    
    def handle_error(self, error: Exception, context: str) -> Any:
        """重试执行"""
        import time
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"重试执行 [{context}] 尝试 {attempt + 1}/{self.max_retries}")
                time.sleep(self.delay)
                # 这里需要具体的重试逻辑
                # 实际实现中需要传入可调用的函数
                break
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"重试失败 [{context}]: {e}")
                    raise
                logger.warning(f"重试失败 [{context}] 尝试 {attempt + 1}: {e}")


class GracefulDegradationStrategy(ErrorHandlingStrategy):
    """优雅降级策略"""
    
    def __init__(self, degraded_services: Dict[str, Any]):
        self.degraded_services = degraded_services
    
    def handle_error(self, error: Exception, context: str) -> Any:
        """使用降级服务"""
        logger.warning(f"服务降级 [{context}]: {error}")
        return self.degraded_services.get(context)


class SystemOrchestrator:
    """系统编排器"""
    
    def __init__(self, dependency_injector: DependencyInjector):
        self.dependency_injector = dependency_injector
        self.modules: Dict[str, ModuleInterface] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_module(self, module_config: ModuleConfig):
        """注册模块"""
        try:
            module = self.dependency_injector.create_module(module_config)
            self.modules[module_config.name] = module
            self.logger.info(f"注册模块: {module_config.name}")
        except Exception as e:
            self.logger.error(f"注册模块失败 {module_config.name}: {e}")
            raise
    
    def start_system(self):
        """启动系统"""
        self.logger.info("启动系统...")
        
        for name, module in self.modules.items():
            try:
                self.logger.info(f"启动模块: {name}")
                # 模块启动逻辑（如果需要）
            except Exception as e:
                self.logger.error(f"启动模块失败 {name}: {e}")
                raise
    
    def stop_system(self):
        """停止系统"""
        self.logger.info("停止系统...")
        
        for name, module in self.modules.items():
            try:
                self.logger.info(f"停止模块: {name}")
                module.shutdown()
            except Exception as e:
                self.logger.error(f"停止模块失败 {name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            "system": "running",
            "modules": {},
            "timestamp": time.time()
        }
        
        for name, module in self.modules.items():
            try:
                status["modules"][name] = module.get_status()
            except Exception as e:
                status["modules"][name] = {"status": "error", "error": str(e)}
        
        return status


# 全局依赖注入容器
_global_service_locator = ServiceLocator()
_global_config_manager = None
_global_orchestrator = None


def initialize_dependency_injection(base_path: Optional[str] = None) -> SystemOrchestrator:
    """初始化全局依赖注入"""
    global _global_service_locator, _global_config_manager, _global_orchestrator
    
    # 初始化路径解析器
    path_resolver = PathResolver(base_path)
    
    # 设置默认路径映射
    path_resolver.add_mapping("project_root", ".")
    path_resolver.add_mapping("modules", "modules")
    path_resolver.add_mapping("config", "config")
    path_resolver.add_mapping("data", "data")
    path_resolver.add_mapping("logs", "logs")
    path_resolver.add_mapping("models", "models")
    path_resolver.add_mapping("services", "services")
    
    # 初始化配置管理器
    _global_config_manager = ConfigManager(path_resolver)
    
    # 初始化依赖注入器
    dependency_injector = DependencyInjector(_global_service_locator, _global_config_manager)
    
    # 初始化系统编排器
    _global_orchestrator = SystemOrchestrator(dependency_injector)
    
    logger.info("依赖注入系统初始化完成")
    return _global_orchestrator


def get_service_locator() -> ServiceLocator:
    """获取全局服务定位器"""
    if _global_service_locator is None:
        raise RuntimeError("依赖注入系统未初始化")
    return _global_service_locator


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    if _global_config_manager is None:
        raise RuntimeError("依赖注入系统未初始化")
    return _global_config_manager


def get_orchestrator() -> SystemOrchestrator:
    """获取全局系统编排器"""
    if _global_orchestrator is None:
        raise RuntimeError("依赖注入系统未初始化")
    return _global_orchestrator


def register_service(config: ModuleConfig):
    """注册服务（全局）"""
    get_service_locator().register(config)


def get_service(name: str, *args, **kwargs) -> Any:
    """获取服务（全局）"""
    return get_service_locator().get(name, *args, **kwargs)


def load_config(name: str, config_path: str, config_type: str = "yaml"):
    """加载配置（全局）"""
    get_config_manager().load_config(name, config_path, config_type)


if __name__ == "__main__":
    # 示例用法
    import time
    
    # 初始化依赖注入系统
    orchestrator = initialize_dependency_injection("/workspace/code")
    
    # 注册服务配置
    service_config = ModuleConfig(
        name="traffic_predictor",
        class_path="models.congestion_predictor.CongestionPropagationPredictor",
        config={
            "input_dim": 4,
            "hidden_dim": 64,
            "output_dim": 3
        },
        singleton=True
    )
    
    register_service(service_config)
    
    # 启动系统
    orchestrator.start_system()
    
    # 获取服务
    try:
        predictor = get_service("traffic_predictor")
        print(f"获取到预测器: {type(predictor)}")
    except Exception as e:
        print(f"获取服务失败: {e}")
    
    # 获取系统状态
    status = orchestrator.get_system_status()
    print(f"系统状态: {status}")
    
    # 停止系统
    orchestrator.stop_system()