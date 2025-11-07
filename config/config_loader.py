"""
配置加载器模块

此模块负责加载、解析和验证YAML格式的配置文件，支持环境变量替换、变量插值
和配置验证功能，确保实验配置的一致性和可复现性。
"""

import os
import yaml
import json
import copy
from datetime import datetime
from typing import Dict, Any, Optional, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('config_loader')

class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self):
        """初始化配置加载器"""
        self.config = {}
        self.raw_config = {}
        logger.info("配置加载器初始化完成")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            加载的配置字典
        """
        logger.info(f"加载配置文件: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                # 保存原始配置
                self.raw_config = yaml.safe_load(f)
                
                # 深拷贝以避免修改原始配置
                self.config = copy.deepcopy(self.raw_config)
                
                # 执行变量替换
                self._resolve_variables()
                
                # 验证配置
                self._validate_config()
                
                logger.info("配置文件加载成功")
                return self.config
                
        except yaml.YAMLError as e:
            logger.error(f"解析YAML配置失败: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            raise
    
    def _resolve_variables(self):
        """
        解析配置中的变量替换
        支持的变量类型：
        - ${now}: 当前时间
        - ${env:VAR_NAME}: 环境变量
        - ${config:path.to.value}: 配置内部引用
        """
        logger.info("解析配置变量...")
        self.config = self._resolve_recursive(self.config)
    
    def _resolve_recursive(self, obj: Any) -> Any:
        """
        递归解析变量
        
        Args:
            obj: 要解析的对象
            
        Returns:
            解析后的对象
        """
        if isinstance(obj, str):
            # 处理时间变量
            if '${now}' in obj:
                now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                obj = obj.replace('${now}', now_str)
            
            # 处理环境变量
            while '${env:' in obj:
                start = obj.find('${env:')
                end = obj.find('}', start)
                if end == -1:
                    break
                
                var_name = obj[start + 6:end]
                var_value = os.environ.get(var_name, '')
                obj = obj[:start] + var_value + obj[end + 1:]
            
            # 处理配置内部引用
            while '${config:' in obj:
                start = obj.find('${config:')
                end = obj.find('}', start)
                if end == -1:
                    break
                
                path = obj[start + 9:end]
                value = self._get_config_value(self.raw_config, path)
                obj = obj[:start] + str(value) + obj[end + 1:]
            
        elif isinstance(obj, dict):
            return {key: self._resolve_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_recursive(item) for item in obj]
        
        return obj
    
    def _get_config_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        根据路径获取配置值
        
        Args:
            config: 配置字典
            path: 配置路径，如 "model.num_nodes"
            
        Returns:
            配置值
        """
        parts = path.split('.')
        value = config
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            logger.warning(f"配置路径不存在: {path}")
            return ''
    
    def _validate_config(self):
        """
        验证配置的有效性
        """
        logger.info("验证配置有效性...")
        
        # 检查必要的配置部分
        required_sections = ['metadata', 'dataset', 'model', 'training', 'evaluation']
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"配置缺少必要部分: {section}")
        
        # 验证数据集名称
        if 'dataset' in self.config and 'name' in self.config['dataset']:
            valid_datasets = ['METR-LA', 'PEMS-BAY']
            if self.config['dataset']['name'] not in valid_datasets:
                logger.warning(f"数据集名称无效: {self.config['dataset']['name']}，应从 {valid_datasets} 中选择")
        
        # 验证随机种子
        if 'metadata' in self.config and 'random_seed' in self.config['metadata']:
            seed = self.config['metadata']['random_seed']
            if not isinstance(seed, int) or seed < 0:
                logger.warning(f"随机种子无效: {seed}，应是非负整数")
        
        # 验证训练集比例
        if 'dataset' in self.config:
            train_ratio = self.config['dataset'].get('train_ratio', 0.7)
            val_ratio = self.config['dataset'].get('val_ratio', 0.1)
            test_ratio = self.config['dataset'].get('test_ratio', 0.2)
            
            if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
                logger.warning(f"数据集比例不匹配: train={train_ratio}, val={val_ratio}, test={test_ratio}，总和应为1.0")
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            path: 配置路径，如 "model.num_nodes"
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        try:
            value = self._get_config_value(self.config, path)
            return value if value != '' else default
        except (KeyError, TypeError):
            return default
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 要更新的配置字典
        """
        logger.info("更新配置...")
        self._update_recursive(self.config, updates)
        logger.info("配置更新完成")
    
    def _update_recursive(self, config: Dict[str, Any], updates: Dict[str, Any]):
        """
        递归更新配置
        
        Args:
            config: 原始配置
            updates: 要更新的配置
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_recursive(config[key], value)
            else:
                config[key] = value
    
    def save_config(self, output_path: str):
        """
        保存配置到文件
        
        Args:
            output_path: 输出文件路径
        """
        logger.info(f"保存配置到: {output_path}")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            # 保存为YAML格式
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            # 同时保存为JSON格式（便于程序读取）
            json_path = output_path.replace('.yaml', '.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置保存成功: {output_path}, {json_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def get_experiment_config(self, experiment_type: str = None) -> Dict[str, Any]:
        """
        获取特定类型实验的配置
        
        Args:
            experiment_type: 实验类型，如 'ablation', 'baseline'
            
        Returns:
            实验配置
        """
        if experiment_type == 'ablation' and 'ablation' in self.config:
            return self.config['ablation']
        elif experiment_type == 'baseline' and 'baseline_models' in self.config.get('evaluation', {}):
            return {'baseline_models': self.config['evaluation']['baseline_models']}
        else:
            return self.config
    
    def create_ablation_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        创建消融实验配置
        
        Returns:
            消融实验配置字典
        """
        ablation_configs = {}
        
        if 'ablation' in self.config and self.config['ablation'].get('enabled', False):
            base_config = copy.deepcopy(self.config)
            experiments = self.config['ablation'].get('experiments', [])
            
            for exp in experiments:
                exp_name = exp.get('name')
                exp_mods = exp.get('modifications', {})
                
                if exp_name:
                    # 创建实验配置副本
                    exp_config = copy.deepcopy(base_config)
                    
                    # 应用修改
                    for key_path, value in exp_mods.items():
                        keys = key_path.split('.')
                        target = exp_config
                        
                        # 导航到目标位置
                        for key in keys[:-1]:
                            if key not in target:
                                target[key] = {}
                            target = target[key]
                        
                        # 设置值
                        target[keys[-1]] = value
                    
                    # 更新实验名称
                    exp_config['metadata']['experiment_name'] = f"{base_config['metadata']['experiment_name']}_ablation_{exp_name}"
                    
                    ablation_configs[exp_name] = exp_config
                    logger.info(f"创建消融实验配置: {exp_name}")
        
        return ablation_configs
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        获取模型配置
        
        Returns:
            模型配置字典
        """
        model_config = self.config.get('model', {})
        
        # 添加随机种子
        if 'metadata' in self.config:
            model_config['random_seed'] = self.config['metadata'].get('random_seed', 42)
        
        return model_config
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        获取训练配置
        
        Returns:
            训练配置字典
        """
        training_config = self.config.get('training', {})
        
        # 添加数据集相关配置
        if 'dataset' in self.config:
            training_config['batch_size'] = self.config['dataset'].get('batch_size', 32)
        
        return training_config
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """
        获取评估配置
        
        Returns:
            评估配置字典
        """
        return self.config.get('evaluation', {})

def load_config(config_path: str) -> Dict[str, Any]:
    """
    便捷函数：加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    loader = ConfigLoader()
    return loader.load_config(config_path)

def create_config_template(template_type: str = 'default') -> Dict[str, Any]:
    """
    创建配置模板
    
    Args:
        template_type: 模板类型
        
    Returns:
        配置模板字典
    """
    templates = {
        'default': {
            'metadata': {
                'experiment_name': 'default_experiment',
                'description': '',
                'author': '',
                'creation_date': '${now}',
                'random_seed': 42
            },
            'dataset': {
                'name': 'METR-LA',
                'data_dir': '',
                'sequence_length': 12,
                'prediction_steps': 3,
                'batch_size': 32
            },
            'model': {},
            'training': {},
            'evaluation': {}
        },
        'ablation': {
            'metadata': {
                'experiment_name': 'ablation_experiment',
                'description': '消融实验',
                'random_seed': 42
            },
            'ablation': {
                'enabled': True,
                'experiments': [
                    {
                        'name': 'baseline',
                        'description': '基线实验',
                        'modifications': {}
                    }
                ]
            }
        },
        'baseline': {
            'evaluation': {
                'baseline_models': [
                    {
                        'name': 'lstm',
                        'params': {}
                    },
                    {
                        'name': 'gcn',
                        'params': {}
                    }
                ]
            }
        }
    }
    
    return templates.get(template_type, templates['default'])

def save_config_template(output_path: str, template_type: str = 'default'):
    """
    保存配置模板
    
    Args:
        output_path: 输出路径
        template_type: 模板类型
    """
    template = create_config_template(template_type)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"配置模板保存到: {output_path}")

if __name__ == "__main__":
    # 示例用法
    try:
        # 加载配置
        loader = ConfigLoader()
        
        # 创建并保存模板
        template_dir = "d:/gcn-lstm/configs/templates"
        save_config_template(os.path.join(template_dir, "default.yaml"), 'default')
        save_config_template(os.path.join(template_dir, "ablation.yaml"), 'ablation')
        save_config_template(os.path.join(template_dir, "baseline.yaml"), 'baseline')
        
        print("配置模板创建完成")
        
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()