#!/usr/bin/env python3
"""
配置管理模块
管理数据源配置、系统参数和API密钥
"""

import os
import sys
import json
import yaml
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading

# 添加项目路径
sys.path.append('/workspace/code')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API配置"""
    name: str
    api_key: str
    base_url: str
    enabled: bool = True
    rate_limit: int = 100  # 每分钟请求限制
    timeout: int = 30
    retry_times: int = 3
    retry_delay: float = 1.0

@dataclass
class DataSourceConfig:
    """数据源配置"""
    name: str
    source_type: str  # pems, gaode, baidu, heweather
    enabled: bool = True
    priority: int = 1  # 优先级，数字越小优先级越高
    api_config: Optional[APIConfig] = None
    fallback_enabled: bool = True
    cache_enabled: bool = True
    cache_timeout: int = 300  # 缓存超时时间(秒)

@dataclass
class QualityConfig:
    """数据质量配置"""
    min_completeness: float = 0.80
    min_accuracy: float = 0.80
    min_timeliness: float = 0.70
    outlier_std_threshold: float = 3.0
    max_data_age_minutes: int = 30
    monitoring_enabled: bool = True
    alert_enabled: bool = True

@dataclass
class SystemConfig:
    """系统配置"""
    log_level: str = "INFO"
    log_file: str = "/workspace/logs/traffic_system.log"
    data_dir: str = "/workspace/data"
    cache_dir: str = "/workspace/cache"
    temp_dir: str = "/workspace/tmp"
    max_workers: int = 4
    request_timeout: int = 30
    database_url: str = "/workspace/data/traffic_system.db"

@dataclass
class AppConfig:
    """应用主配置"""
    system: SystemConfig
    data_sources: List[DataSourceConfig]
    quality: QualityConfig
    version: str = "2.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "/workspace/config/real_data_config.yaml"):
        self.config_file = config_file
        self.config_lock = threading.Lock()
        self._ensure_config_dir()
        self.config = self._load_config()
        
        logger.info(f"配置管理器初始化完成，配置文件: {config_file}")
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        config_dir = os.path.dirname(self.config_file)
        os.makedirs(config_dir, exist_ok=True)
        
        # 确保日志目录存在
        log_dir = os.path.dirname("/workspace/logs/traffic_system.log")
        os.makedirs(log_dir, exist_ok=True)
    
    def _load_config(self) -> AppConfig:
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # 重建配置对象
                return self._dict_to_config(config_data)
            else:
                logger.info("配置文件不存在，使用默认配置")
                return self._create_default_config()
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._create_default_config()
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """将字典转换为配置对象"""
        try:
            # 系统配置
            system_config = SystemConfig(**data.get('system', {}))
            
            # 数据源配置
            data_sources = []
            for ds_data in data.get('data_sources', []):
                api_config = None
                if 'api_config' in ds_data and ds_data['api_config']:
                    api_config = APIConfig(**ds_data['api_config'])
                
                data_source = DataSourceConfig(
                    name=ds_data['name'],
                    source_type=ds_data['source_type'],
                    enabled=ds_data.get('enabled', True),
                    priority=ds_data.get('priority', 1),
                    api_config=api_config,
                    fallback_enabled=ds_data.get('fallback_enabled', True),
                    cache_enabled=ds_data.get('cache_enabled', True),
                    cache_timeout=ds_data.get('cache_timeout', 300)
                )
                data_sources.append(data_source)
            
            # 质量配置
            quality_config = QualityConfig(**data.get('quality', {}))
            
            # 应用配置
            app_config = AppConfig(
                system=system_config,
                data_sources=data_sources,
                quality=quality_config,
                version=data.get('version', '2.0.0'),
                created_at=data.get('created_at', datetime.now().isoformat()),
                updated_at=datetime.now().isoformat()
            )
            
            return app_config
            
        except Exception as e:
            logger.error(f"转换配置字典失败: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> AppConfig:
        """创建默认配置"""
        # 系统配置
        system_config = SystemConfig()
        
        # API配置
        gaode_api = APIConfig(
            name="高德地图API",
            api_key="c2331ea112926323b618dfbd40168aa1",
            base_url="https://restapi.amap.com/v3",
            rate_limit=100,
            timeout=30
        )
        
        baidu_api = APIConfig(
            name="百度地图API",
            api_key="sLAWylWoVFbPfHLFDr8t9v4iyyMrDGuI",
            base_url="https://api.map.baidu.com",
            rate_limit=100,
            timeout=30
        )
        
        heweather_api = APIConfig(
            name="和风天气API",
            api_key="ccf9fb7e26eb48b99ea87f363a6c79b3",
            base_url="https://devapi.qweather.com/v7",
            rate_limit=1000,
            timeout=30
        )
        
        # 数据源配置
        data_sources = [
            DataSourceConfig(
                name="PEMS数据集",
                source_type="pems",
                enabled=True,
                priority=1,
                cache_enabled=True,
                cache_timeout=600
            ),
            DataSourceConfig(
                name="高德地图",
                source_type="gaode",
                enabled=True,
                priority=2,
                api_config=gaode_api,
                cache_enabled=True,
                cache_timeout=300
            ),
            DataSourceConfig(
                name="百度地图",
                source_type="baidu",
                enabled=True,
                priority=3,
                api_config=baidu_api,
                cache_enabled=True,
                cache_timeout=300
            ),
            DataSourceConfig(
                name="和风天气",
                source_type="heweather",
                enabled=True,
                priority=1,
                api_config=heweather_api,
                cache_enabled=True,
                cache_timeout=1800  # 天气数据缓存时间较长
            )
        ]
        
        # 质量配置
        quality_config = QualityConfig()
        
        # 应用配置
        app_config = AppConfig(
            system=system_config,
            data_sources=data_sources,
            quality=quality_config
        )
        
        # 保存默认配置
        self.save_config(app_config)
        
        return app_config
    
    def save_config(self, config: AppConfig = None):
        """保存配置到文件"""
        try:
            with self.config_lock:
                if config is None:
                    config = self.config
                
                config.updated_at = datetime.now().isoformat()
                
                # 转换为字典
                config_dict = self._config_to_dict(config)
                
                # 保存到文件
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                
                logger.info(f"配置已保存到: {self.config_file}")
                
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        config_dict = {
            'version': config.version,
            'created_at': config.created_at,
            'updated_at': config.updated_at,
            'system': asdict(config.system),
            'data_sources': [],
            'quality': asdict(config.quality)
        }
        
        # 转换数据源配置
        for ds in config.data_sources:
            ds_dict = {
                'name': ds.name,
                'source_type': ds.source_type,
                'enabled': ds.enabled,
                'priority': ds.priority,
                'fallback_enabled': ds.fallback_enabled,
                'cache_enabled': ds.cache_enabled,
                'cache_timeout': ds.cache_timeout
            }
            
            if ds.api_config:
                ds_dict['api_config'] = asdict(ds.api_config)
            
            config_dict['data_sources'].append(ds_dict)
        
        return config_dict
    
    def get_config(self) -> AppConfig:
        """获取当前配置"""
        return self.config
    
    def update_data_source(self, source_name: str, updates: Dict[str, Any]):
        """更新数据源配置"""
        try:
            with self.config_lock:
                for data_source in self.config.data_sources:
                    if data_source.name == source_name:
                        # 更新字段
                        for key, value in updates.items():
                            if hasattr(data_source, key):
                                setattr(data_source, key, value)
                        
                        self.save_config()
                        logger.info(f"数据源 {source_name} 配置已更新")
                        return True
                
                logger.warning(f"未找到数据源: {source_name}")
                return False
                
        except Exception as e:
            logger.error(f"更新数据源配置失败: {e}")
            return False
    
    def enable_data_source(self, source_name: str):
        """启用数据源"""
        return self.update_data_source(source_name, {'enabled': True})
    
    def disable_data_source(self, source_name: str):
        """禁用数据源"""
        return self.update_data_source(source_name, {'enabled': False})
    
    def get_enabled_data_sources(self) -> List[DataSourceConfig]:
        """获取启用的数据源"""
        enabled_sources = [ds for ds in self.config.data_sources if ds.enabled]
        # 按优先级排序
        enabled_sources.sort(key=lambda x: x.priority)
        return enabled_sources
    
    def get_data_source_by_type(self, source_type: str) -> Optional[DataSourceConfig]:
        """根据类型获取数据源"""
        for data_source in self.config.data_sources:
            if data_source.source_type == source_type:
                return data_source
        return None
    
    def update_api_key(self, source_type: str, api_key: str):
        """更新API密钥"""
        try:
            with self.config_lock:
                data_source = self.get_data_source_by_type(source_type)
                if data_source and data_source.api_config:
                    data_source.api_config.api_key = api_key
                    self.save_config()
                    logger.info(f"API密钥已更新: {source_type}")
                    return True
                else:
                    logger.warning(f"未找到数据源或API配置: {source_type}")
                    return False
                    
        except Exception as e:
            logger.error(f"更新API密钥失败: {e}")
            return False
    
    def get_quality_config(self) -> QualityConfig:
        """获取质量配置"""
        return self.config.quality
    
    def update_quality_config(self, updates: Dict[str, Any]):
        """更新质量配置"""
        try:
            with self.config_lock:
                for key, value in updates.items():
                    if hasattr(self.config.quality, key):
                        setattr(self.config.quality, key, value)
                
                self.save_config()
                logger.info("质量配置已更新")
                
        except Exception as e:
            logger.error(f"更新质量配置失败: {e}")
    
    def get_system_config(self) -> SystemConfig:
        """获取系统配置"""
        return self.config.system
    
    def validate_config(self) -> List[str]:
        """验证配置"""
        errors = []
        
        try:
            # 验证API密钥
            for data_source in self.config.data_sources:
                if data_source.enabled and data_source.api_config:
                    if not data_source.api_config.api_key:
                        errors.append(f"数据源 {data_source.name} 缺少API密钥")
                    
                    if not data_source.api_config.base_url:
                        errors.append(f"数据源 {data_source.name} 缺少基础URL")
            
            # 验证质量配置
            quality = self.config.quality
            if not (0 <= quality.min_completeness <= 1):
                errors.append("完整性阈值必须在0-1之间")
            
            if not (0 <= quality.min_accuracy <= 1):
                errors.append("准确性阈值必须在0-1之间")
            
            if quality.max_data_age_minutes <= 0:
                errors.append("最大数据年龄必须大于0")
            
            # 验证系统配置
            system = self.config.system
            if system.max_workers <= 0:
                errors.append("最大工作线程数必须大于0")
            
            if system.request_timeout <= 0:
                errors.append("请求超时时间必须大于0")
            
            # 检查目录是否存在
            for dir_path in [system.data_dir, system.cache_dir, system.temp_dir]:
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path, exist_ok=True)
                    except Exception as e:
                        errors.append(f"无法创建目录 {dir_path}: {e}")
            
        except Exception as e:
            errors.append(f"配置验证过程出错: {e}")
        
        return errors
    
    def export_config(self, export_file: str):
        """导出配置"""
        try:
            config_dict = self._config_to_dict(self.config)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已导出到: {export_file}")
            
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
    
    def import_config(self, import_file: str):
        """导入配置"""
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            new_config = self._dict_to_config(config_data)
            self.config = new_config
            self.save_config()
            
            logger.info(f"配置已从 {import_file} 导入")
            
        except Exception as e:
            logger.error(f"导入配置失败: {e}")
    
    def reset_to_default(self):
        """重置为默认配置"""
        try:
            self.config = self._create_default_config()
            logger.info("配置已重置为默认值")
            
        except Exception as e:
            logger.error(f"重置配置失败: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        try:
            summary = {
                'version': self.config.version,
                'created_at': self.config.created_at,
                'updated_at': self.config.updated_at,
                'data_sources': {
                    'total': len(self.config.data_sources),
                    'enabled': len([ds for ds in self.config.data_sources if ds.enabled]),
                    'disabled': len([ds for ds in self.config.data_sources if not ds.enabled])
                },
                'quality_config': {
                    'min_completeness': self.config.quality.min_completeness,
                    'min_accuracy': self.config.quality.min_accuracy,
                    'min_timeliness': self.config.quality.min_timeliness,
                    'monitoring_enabled': self.config.quality.monitoring_enabled
                },
                'system_config': {
                    'log_level': self.config.system.log_level,
                    'max_workers': self.config.system.max_workers,
                    'request_timeout': self.config.system.request_timeout
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"生成配置摘要失败: {e}")
            return {'error': str(e)}

# 全局配置管理器实例
config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager

# 使用示例
if __name__ == "__main__":
    # 初始化配置管理器
    manager = ConfigManager()
    
    # 获取配置摘要
    print("配置摘要:")
    summary = manager.get_config_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    # 获取启用的数据源
    print("\n启用的数据源:")
    enabled_sources = manager.get_enabled_data_sources()
    for source in enabled_sources:
        print(f"- {source.name} ({source.source_type}) - 优先级: {source.priority}")
    
    # 验证配置
    print("\n配置验证:")
    errors = manager.validate_config()
    if errors:
        print("发现错误:")
        for error in errors:
            print(f"- {error}")
    else:
        print("配置验证通过")
    
    # 更新API密钥示例
    print("\n更新API密钥示例:")
    success = manager.update_api_key("gaode", "new_api_key_here")
    print(f"更新结果: {success}")
    
    # 导出配置示例
    print("\n导出配置示例:")
    manager.export_config("/workspace/config/exported_config.json")
    
    # 更新质量配置示例
    print("\n更新质量配置示例:")
    manager.update_quality_config({
        'min_completeness': 0.85,
        'max_data_age_minutes': 20
    })
    
    print("配置管理演示完成")