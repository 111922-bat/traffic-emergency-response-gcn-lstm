#!/usr/bin/env python3
"""
重构后的交通流预测模块示例
Refactored Traffic Flow Prediction Module Example

展示如何使用新的依赖注入、配置管理、接口抽象和错误处理系统
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

# 添加重构架构路径
sys.path.append(str(Path(__file__).parent))

from dependency_injection import (
    initialize_dependency_injection, register_service, get_service, 
    ModuleConfig, get_config_manager, get_orchestrator
)
from modular_config import initialize_config_manager, load_module_config
from module_interfaces import (
    BaseModule, TrafficDataInterface, PredictionModelInterface,
    TrafficDataModule, PredictionModelModule, ModuleStatus
)
from error_handling import (
    initialize_error_handling, error_handler, graceful_degradation,
    ErrorCategory, ErrorSeverity, get_global_error_handler
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RefactoredTrafficPredictor(BaseModule):
    """重构后的交通流预测器"""
    
    def __init__(self):
        super().__init__("refactored_traffic_predictor", "2.0.0")
        self.add_capability("traffic_prediction")
        self.add_capability("real_time_analysis")
        self.add_capability("batch_processing")
        
        # 依赖服务
        self.traffic_data_service = None
        self.prediction_model_service = None
        self.config_manager = None
        
        # 配置
        self.prediction_horizon = 6
        self.update_interval = 5
        self.confidence_threshold = 0.8
        
        # 状态
        self.is_initialized = False
        self.last_prediction_time = None
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化模块"""
        try:
            self.logger.info("初始化重构后的交通流预测器")
            
            # 获取依赖服务
            self.traffic_data_service = get_service("traffic_data")
            self.prediction_model_service = get_service("prediction_model")
            self.config_manager = get_config_manager()
            
            # 加载配置
            await self._load_configuration(config)
            
            self.is_initialized = True
            self.status = ModuleStatus.READY
            self.logger.info("重构后的交通流预测器初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def _load_configuration(self, config: Dict[str, Any]):
        """加载配置"""
        # 从配置管理器加载模块配置
        module_config = self.config_manager.get_module_config("traffic_predictor")
        
        # 更新配置
        self.prediction_horizon = module_config.get("prediction_horizon", 6)
        self.update_interval = module_config.get("update_interval", 5)
        self.confidence_threshold = module_config.get("confidence_threshold", 0.8)
        
        self.logger.info(f"加载配置: horizon={self.prediction_horizon}, "
                        f"interval={self.update_interval}, "
                        f"threshold={self.confidence_threshold}")
    
    async def start(self) -> bool:
        """启动模块"""
        try:
            self.logger.info("启动重构后的交通流预测器")
            self.status = ModuleStatus.RUNNING
            self._update_last_active()
            return True
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """停止模块"""
        try:
            self.logger.info("停止重构后的交通流预测器")
            self.status = ModuleStatus.STOPPED
            return True
        except Exception as e:
            self.logger.error(f"停止失败: {e}")
            return False
    
    async def pause(self) -> bool:
        """暂停模块"""
        self.status = ModuleStatus.PAUSED
        self.logger.info("暂停重构后的交通流预测器")
        return True
    
    async def resume(self) -> bool:
        """恢复模块"""
        self.status = ModuleStatus.RUNNING
        self.logger.info("恢复重构后的交通流预测器")
        return True
    
    def get_status(self) -> ModuleStatus:
        """获取模块状态"""
        return self.status
    
    @error_handler(
        error_category=ErrorCategory.BUSINESS_LOGIC,
        error_severity=ErrorSeverity.MEDIUM,
        fallback_return={"status": "error", "message": "预测服务不可用"}
    )
    @graceful_degradation("traffic_prediction", lambda: {"status": "degraded", "message": "使用降级预测"})
    async def predict_traffic_flow(self, location_ids: List[str], 
                                 prediction_horizon: Optional[int] = None) -> Dict[str, Any]:
        """预测交通流"""
        if not self.is_initialized:
            raise RuntimeError("模块未初始化")
        
        self._update_last_active()
        horizon = prediction_horizon or self.prediction_horizon
        
        try:
            # 获取当前交通数据
            current_data = await self.traffic_data_service.get_current_traffic_data(location_ids)
            
            # 执行预测
            prediction_input = {
                "locations": location_ids,
                "current_data": current_data,
                "horizon": horizon,
                "timestamp": datetime.now().isoformat()
            }
            
            prediction_result = await self.prediction_model_service.predict(prediction_input, horizon)
            
            # 后处理
            processed_result = await self._post_process_prediction(prediction_result)
            
            self.last_prediction_time = datetime.now()
            self.logger.info(f"完成交通流预测: {len(location_ids)} 个位置, {horizon} 步预测")
            
            return {
                "status": "success",
                "prediction_id": prediction_result.get("prediction_id"),
                "locations": location_ids,
                "horizon": horizon,
                "results": processed_result,
                "timestamp": self.last_prediction_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"交通流预测失败: {e}")
            raise
    
    async def _post_process_prediction(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """后处理预测结果"""
        results = prediction_result.get("results", {})
        
        # 置信度过滤
        if "confidence" in results:
            confidence_scores = results["confidence"]
            high_confidence_mask = [score >= self.confidence_threshold for score in confidence_scores]
            results["high_confidence_mask"] = high_confidence_mask
        
        # 添加元数据
        results["metadata"] = {
            "confidence_threshold": self.confidence_threshold,
            "processing_time": datetime.now().isoformat(),
            "model_version": "2.0.0"
        }
        
        return results
    
    @error_handler(
        error_category=ErrorCategory.DATABASE,
        error_severity=ErrorSeverity.HIGH,
        fallback_return=[]
    )
    async def get_real_time_predictions(self, location_ids: List[str]) -> List[Dict[str, Any]]:
        """获取实时预测"""
        if not self.is_initialized:
            raise RuntimeError("模块未初始化")
        
        self._update_last_active()
        
        try:
            # 检查是否需要更新预测
            if (self.last_prediction_time is None or 
                datetime.now() - self.last_prediction_time > timedelta(seconds=self.update_interval)):
                
                # 执行新的预测
                prediction_result = await self.predict_traffic_flow(location_ids)
                return [prediction_result]
            else:
                # 返回缓存的预测（简化实现）
                return [{
                    "status": "cached",
                    "message": "使用缓存的预测结果",
                    "last_update": self.last_prediction_time.isoformat()
                }]
                
        except Exception as e:
            self.logger.error(f"获取实时预测失败: {e}")
            raise
    
    async def batch_predict(self, prediction_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量预测"""
        if not self.is_initialized:
            raise RuntimeError("模块未初始化")
        
        self._update_last_active()
        
        try:
            # 准备批量输入
            batch_input = {
                "requests": prediction_requests,
                "timestamp": datetime.now().isoformat()
            }
            
            # 执行批量预测
            batch_result = await self.prediction_model_service.predict_batch(
                [req["input_data"] for req in prediction_requests],
                self.prediction_horizon
            )
            
            # 组合结果
            results = []
            for i, (request, prediction) in enumerate(zip(prediction_requests, batch_result)):
                results.append({
                    "request_id": request.get("request_id", f"req_{i}"),
                    "status": "success",
                    "prediction": prediction,
                    "timestamp": datetime.now().isoformat()
                })
            
            self.logger.info(f"完成批量预测: {len(prediction_requests)} 个请求")
            return results
            
        except Exception as e:
            self.logger.error(f"批量预测失败: {e}")
            raise
    
    def get_module_statistics(self) -> Dict[str, Any]:
        """获取模块统计信息"""
        return {
            "module_name": self.name,
            "version": self.version,
            "status": self.status.value,
            "capabilities": self._capabilities,
            "is_initialized": self.is_initialized,
            "last_prediction_time": self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            "configuration": {
                "prediction_horizon": self.prediction_horizon,
                "update_interval": self.update_interval,
                "confidence_threshold": self.confidence_threshold
            },
            "dependencies": {
                "traffic_data_service": self.traffic_data_service is not None,
                "prediction_model_service": self.prediction_model_service is not None,
                "config_manager": self.config_manager is not None
            },
            "timestamp": datetime.now().isoformat()
        }


class SystemOrchestrator:
    """系统编排器 - 展示重构后的系统集成"""
    
    def __init__(self):
        self.modules = {}
        self.logger = logging.getLogger(__name__)
    
    async def initialize_system(self, config_dir: str = "config") -> bool:
        """初始化系统"""
        try:
            self.logger.info("开始初始化重构后的系统")
            
            # 1. 初始化配置管理
            config_manager = initialize_config_manager(config_dir)
            
            # 2. 初始化依赖注入
            orchestrator = initialize_dependency_injection()
            
            # 3. 初始化错误处理
            error_handler = initialize_error_handling()
            
            # 4. 加载模块配置
            await self._load_module_configs(config_manager)
            
            # 5. 注册服务
            await self._register_services()
            
            # 6. 创建和注册模块
            await self._create_modules()
            
            self.logger.info("系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统初始化失败: {e}")
            return False
    
    async def _load_module_configs(self, config_manager):
        """加载模块配置"""
        # 交通数据模块配置
        traffic_data_config = {
            "data_source": "mock",
            "update_interval": 5,
            "max_locations": 100
        }
        config_manager.save_module_config("traffic_data", traffic_data_config)
        
        # 预测模型模块配置
        prediction_config = {
            "model_type": "gcn_lstm",
            "device": "cpu",
            "batch_size": 32
        }
        config_manager.save_module_config("prediction_model", prediction_config)
        
        # 交通预测器配置
        predictor_config = {
            "prediction_horizon": 6,
            "update_interval": 5,
            "confidence_threshold": 0.8
        }
        config_manager.save_module_config("traffic_predictor", predictor_config)
        
        self.logger.info("模块配置加载完成")
    
    async def _register_services(self):
        """注册服务"""
        # 注册交通数据服务
        traffic_data_config = ModuleConfig(
            name="traffic_data",
            class_path="module_interfaces.TrafficDataModule",
            singleton=True
        )
        register_service(traffic_data_config)
        
        # 注册预测模型服务
        prediction_model_config = ModuleConfig(
            name="prediction_model",
            class_path="module_interfaces.PredictionModelModule",
            singleton=True
        )
        register_service(prediction_model_config)
        
        self.logger.info("服务注册完成")
    
    async def _create_modules(self):
        """创建模块"""
        # 创建重构后的交通流预测器
        predictor = RefactoredTrafficPredictor()
        self.modules["traffic_predictor"] = predictor
        
        self.logger.info("模块创建完成")
    
    async def start_system(self) -> bool:
        """启动系统"""
        try:
            self.logger.info("启动系统")
            
            # 初始化所有模块
            for name, module in self.modules.items():
                self.logger.info(f"初始化模块: {name}")
                success = await module.initialize({})
                if not success:
                    self.logger.error(f"模块初始化失败: {name}")
                    return False
            
            # 启动所有模块
            for name, module in self.modules.items():
                self.logger.info(f"启动模块: {name}")
                success = await module.start()
                if not success:
                    self.logger.error(f"模块启动失败: {name}")
                    return False
            
            self.logger.info("系统启动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"系统启动失败: {e}")
            return False
    
    async def stop_system(self):
        """停止系统"""
        try:
            self.logger.info("停止系统")
            
            for name, module in self.modules.items():
                self.logger.info(f"停止模块: {name}")
                await module.stop()
            
            self.logger.info("系统停止完成")
            
        except Exception as e:
            self.logger.error(f"系统停止失败: {e}")
    
    async def test_system(self) -> Dict[str, Any]:
        """测试系统功能"""
        try:
            self.logger.info("开始系统测试")
            
            # 获取预测器模块
            predictor = self.modules.get("traffic_predictor")
            if not predictor:
                raise RuntimeError("预测器模块未找到")
            
            # 测试数据
            test_locations = ["location_001", "location_002", "location_003"]
            
            # 测试实时预测
            real_time_result = await predictor.get_real_time_predictions(test_locations)
            
            # 测试交通流预测
            flow_prediction = await predictor.predict_traffic_flow(test_locations)
            
            # 测试批量预测
            batch_requests = [
                {"request_id": "batch_001", "input_data": {"locations": test_locations[:2]}},
                {"request_id": "batch_002", "input_data": {"locations": test_locations[2:]}}
            ]
            batch_result = await predictor.batch_predict(batch_requests)
            
            # 获取模块统计
            stats = predictor.get_module_statistics()
            
            test_results = {
                "status": "success",
                "real_time_test": real_time_result,
                "flow_prediction_test": flow_prediction,
                "batch_prediction_test": batch_result,
                "module_statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info("系统测试完成")
            return test_results
            
        except Exception as e:
            self.logger.error(f"系统测试失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            # 获取全局错误处理器状态
            error_handler = get_global_error_handler()
            health_status = error_handler.get_system_health()
            
            # 获取模块状态
            module_statuses = {}
            for name, module in self.modules.items():
                module_statuses[name] = {
                    "status": module.status.value,
                    "info": module.info.__dict__
                }
            
            return {
                "system_status": "running",
                "modules": module_statuses,
                "health": health_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


async def main():
    """主函数 - 演示重构后的系统"""
    print("=" * 60)
    print("重构后的智能交通流预测系统演示")
    print("=" * 60)
    
    # 创建系统编排器
    orchestrator = SystemOrchestrator()
    
    try:
        # 1. 初始化系统
        print("\n1. 初始化系统...")
        success = await orchestrator.initialize_system()
        if not success:
            print("❌ 系统初始化失败")
            return
        print("✅ 系统初始化成功")
        
        # 2. 启动系统
        print("\n2. 启动系统...")
        success = await orchestrator.start_system()
        if not success:
            print("❌ 系统启动失败")
            return
        print("✅ 系统启动成功")
        
        # 3. 测试系统功能
        print("\n3. 测试系统功能...")
        test_results = await orchestrator.test_system()
        
        if test_results["status"] == "success":
            print("✅ 系统测试成功")
            print(f"   - 实时预测测试: {len(test_results['real_time_test'])} 个结果")
            print(f"   - 流量预测测试: {test_results['flow_prediction_test']['status']}")
            print(f"   - 批量预测测试: {len(test_results['batch_prediction_test'])} 个结果")
        else:
            print(f"❌ 系统测试失败: {test_results['error']}")
        
        # 4. 获取系统状态
        print("\n4. 获取系统状态...")
        status = orchestrator.get_system_status()
        print(f"   系统状态: {status['system_status']}")
        print(f"   模块数量: {len(status['modules'])}")
        print(f"   健康状态: {status['health']['degradation_level']}")
        
        # 5. 展示重构优势
        print("\n5. 重构优势展示:")
        print("   ✅ 消除了硬编码路径")
        print("   ✅ 实现了松耦合架构")
        print("   ✅ 添加了依赖注入")
        print("   ✅ 提供了优雅的错误处理")
        print("   ✅ 支持配置热重载")
        print("   ✅ 实现了模块化部署")
        
    except Exception as e:
        print(f"❌ 系统运行异常: {e}")
    
    finally:
        # 6. 清理资源
        print("\n6. 清理资源...")
        await orchestrator.stop_system()
        print("✅ 资源清理完成")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())