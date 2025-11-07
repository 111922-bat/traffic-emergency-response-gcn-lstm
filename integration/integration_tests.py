#!/usr/bin/env python3
"""
智能交通流预测系统集成测试套件
System Integration Test Suite for Intelligent Traffic Flow Prediction System

包含模块间集成测试、API接口测试、数据流验证等
"""

import unittest
import requests
import json
import time
import asyncio
import websockets
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
import pandas as pd

# 添加项目路径
sys.path.append('/workspace/code')
sys.path.append('/workspace/production-system')

# 导入测试模块
try:
    from models.congestion_predictor import CongestionPropagationPredictor, RoadSegment, create_sample_data
    from services.llm_service import LLMService, LLMConfig, LLMProvider
    from pathfinding.emergency_dispatcher import EmergencyDispatcher, EmergencyVehicle, VehicleType, TaskPriority
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    MODELS_AVAILABLE = False

# API测试配置
API_BASE_URL = "http://localhost:3001"
TEST_TIMEOUT = 30


class TestConfig:
    """测试配置类"""
    # 测试数据配置
    TEST_SEGMENTS_COUNT = 10
    TEST_PREDICTION_HORIZON = 6
    TEST_VEHICLES_COUNT = 5
    
    # 性能阈值
    MAX_RESPONSE_TIME = 10.0  # 秒
    MIN_SUCCESS_RATE = 0.95   # 95%
    MAX_ERROR_RATE = 0.05     # 5%


class IntegrationTestResult:
    """集成测试结果"""
    def __init__(self):
        self.test_results = []
        self.performance_metrics = {}
        self.errors = []
        self.warnings = []
    
    def add_result(self, test_name: str, success: bool, message: str = "", duration: float = 0):
        self.test_results.append({
            'test_name': test_name,
            'success': success,
            'message': message,
            'duration': duration,
            'timestamp': time.time()
        })
    
    def add_performance_metric(self, metric_name: str, value: float, threshold: float = None):
        self.performance_metrics[metric_name] = {
            'value': value,
            'threshold': threshold,
            'status': 'PASS' if threshold is None or value <= threshold else 'FAIL'
        }
    
    def get_summary(self):
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['success'])
        failed_tests = total_tests - passed_tests
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'performance_metrics': self.performance_metrics,
            'errors': self.errors,
            'warnings': self.warnings
        }


class ModelIntegrationTest(unittest.TestCase):
    """模型模块集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        if not MODELS_AVAILABLE:
            cls.skipTest(cls, "模型模块不可用")
        
        cls.config = {
            'input_dim': 4,
            'hidden_dim': 64,
            'output_dim': 3,
            'gcn_layers': 3,
            'lstm_layers': 2,
            'dropout': 0.1,
            'fusion_weight': 0.6,
            'input_sequence_length': 12,
            'n_nodes': TestConfig.TEST_SEGMENTS_COUNT
        }
        cls.predictor = CongestionPropagationPredictor(cls.config)
    
    def test_model_initialization(self):
        """测试模型初始化"""
        start_time = time.time()
        
        # 检查模型组件是否正确初始化
        self.assertIsNotNone(self.predictor.gcn_model)
        self.assertIsNotNone(self.predictor.lstm_model)
        self.assertIsNotNone(self.predictor.dynamic_graph)
        self.assertIsNotNone(self.predictor.scaler)
        
        duration = time.time() - start_time
        self.assertLess(duration, 5.0, "模型初始化时间过长")
    
    def test_data_preprocessing(self):
        """测试数据预处理"""
        # 创建测试数据
        test_data = pd.DataFrame({
            'segment_id': [f'seg_{i}' for i in range(TestConfig.TEST_SEGMENTS_COUNT)],
            'speed': np.random.uniform(30, 80, TestConfig.TEST_SEGMENTS_COUNT),
            'flow': np.random.uniform(500, 2000, TestConfig.TEST_SEGMENTS_COUNT),
            'occupancy': np.random.uniform(0.2, 0.8, TestConfig.TEST_SEGMENTS_COUNT),
            'capacity': np.random.uniform(1500, 3000, TestConfig.TEST_SEGMENTS_COUNT)
        })
        
        # 测试数据预处理
        processed_data = self.predictor.prepare_data(test_data)
        
        # 验证输出
        self.assertIsNotNone(processed_data)
        self.assertEqual(len(processed_data.shape), 2, "预处理后的数据维度不正确")
    
    def test_congestion_prediction(self):
        """测试拥堵预测功能"""
        # 创建测试路段
        segments = create_sample_data(n_segments=TestConfig.TEST_SEGMENTS_COUNT)
        
        start_time = time.time()
        
        # 执行预测
        results = self.predictor.predict_congestion_propagation(
            segments, 
            prediction_horizon=TestConfig.TEST_PREDICTION_HORIZON
        )
        
        duration = time.time() - start_time
        
        # 验证结果
        self.assertIsNotNone(results)
        self.assertEqual(len(results), TestConfig.TEST_SEGMENTS_COUNT)
        self.assertLess(duration, TestConfig.MAX_RESPONSE_TIME, "预测响应时间过长")
        
        # 验证预测结果结构
        for result in results:
            self.assertIsNotNone(result.predicted_speeds)
            self.assertIsNotNone(result.predicted_flows)
            self.assertIsNotNone(result.confidence_scores)
    
    def test_bottleneck_identification(self):
        """测试瓶颈路段识别"""
        segments = create_sample_data(n_segments=TestConfig.TEST_SEGMENTS_COUNT)
        
        bottlenecks = self.predictor.identify_bottlenecks(segments)
        
        # 验证瓶颈识别结果
        self.assertIsInstance(bottlenecks, list)
        # 瓶颈路段ID应该在segments中存在
        for bottleneck_id in bottlenecks:
            self.assertTrue(
                any(seg.segment_id == bottleneck_id for seg in segments)
            )


class ServiceIntegrationTest(unittest.TestCase):
    """服务模块集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        if not MODELS_AVAILABLE:
            cls.skipTest(cls, "服务模块不可用")
        
        # 创建LLM服务配置（使用模拟API密钥）
        cls.llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key="sk-test-key-for-integration-testing",
            model="gpt-3.5-turbo",
            base_url="https://api.openai.com/v1",
            timeout=10,
            max_retries=2
        )
        cls.llm_service = LLMService(cls.llm_config)
    
    def test_llm_service_initialization(self):
        """测试LLM服务初始化"""
        self.assertIsNotNone(self.llm_service)
        self.assertIsNotNone(self.llm_service.provider)
        self.assertIsNotNone(self.llm_service.metrics)
    
    def test_prompt_templates(self):
        """测试提示模板功能"""
        # 测试默认模板
        self.assertIn("chat", self.llm_service.prompt_templates)
        
        # 测试模板格式化
        formatted_prompt = self.llm_service.format_prompt(
            "chat", 
            user_input="测试消息"
        )
        self.assertIn("测试消息", formatted_prompt)
    
    def test_performance_metrics(self):
        """测试性能指标"""
        # 获取初始指标
        initial_metrics = self.llm_service.get_metrics()
        self.assertIsNotNone(initial_metrics)
        
        # 重置指标
        self.llm_service.reset_metrics()
        reset_metrics = self.llm_service.get_metrics()
        self.assertEqual(reset_metrics.request_count, 0)


class PathfindingIntegrationTest(unittest.TestCase):
    """路径规划模块集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        if not MODELS_AVAILABLE:
            cls.skipTest(cls, "路径规划模块不可用")
        
        cls.dispatcher = EmergencyDispatcher("测试调度中心")
        
        # 添加测试车辆
        from pathfinding.emergency_dispatcher import Location, EmergencyVehicle, VehicleType
        for i in range(TestConfig.TEST_VEHICLES_COUNT):
            vehicle = EmergencyVehicle(
                vehicle_id=f"TEST_VEHICLE_{i:03d}",
                vehicle_type=VehicleType.MEDICAL,
                current_location=Location(i * 10, i * 20, f"测试位置{i}")
            )
            cls.dispatcher.add_vehicle(vehicle)
    
    def test_dispatcher_initialization(self):
        """测试调度器初始化"""
        self.assertIsNotNone(self.dispatcher)
        self.assertEqual(len(self.dispatcher.vehicles), TestConfig.TEST_VEHICLES_COUNT)
        self.assertIsNotNone(self.dispatcher.optimizer)
    
    def test_vehicle_management(self):
        """测试车辆管理功能"""
        # 测试获取可用车辆
        available_vehicles = self.dispatcher.get_available_vehicles()
        self.assertIsInstance(available_vehicles, list)
        
        # 测试车辆状态检查
        for vehicle_id, vehicle in self.dispatcher.vehicles.items():
            self.assertTrue(vehicle.is_available())
    
    def test_task_dispatch(self):
        """测试任务调度功能"""
        from pathfinding.emergency_dispatcher import EmergencyTask, TaskPriority, Location
        
        # 创建测试任务
        task = EmergencyTask(
            task_id="TEST_TASK_001",
            location=Location(50, 100, "测试事故点"),
            task_type=VehicleType.MEDICAL,
            priority=TaskPriority.HIGH,
            description="测试紧急任务"
        )
        
        # 添加任务
        self.dispatcher.add_task(task)
        
        # 执行调度
        dispatch_success = self.dispatcher.dispatch_task(task.task_id)
        
        # 验证调度结果
        self.assertTrue(dispatch_success)
        self.assertEqual(task.status.value, "已分配")
        self.assertGreater(len(task.assigned_vehicles), 0)
    
    def test_optimization_strategies(self):
        """测试优化策略"""
        # 测试切换策略
        self.dispatcher.optimize_strategy("closest")
        self.assertEqual(self.dispatcher.optimizer.current_strategy, "closest")
        
        self.dispatcher.optimize_strategy("efficiency_based")
        self.assertEqual(self.dispatcher.optimizer.current_strategy, "efficiency_based")
        
        # 测试无效策略
        self.dispatcher.optimize_strategy("invalid_strategy")
        # 应该保持原有策略不变


class APIIntegrationTest(unittest.TestCase):
    """API接口集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.api_available = cls.check_api_availability()
    
    @classmethod
    def check_api_availability(cls):
        """检查API服务是否可用"""
        try:
            response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_api_health_check(self):
        """测试API健康检查"""
        if not self.api_available:
            self.skipTest("API服务不可用")
        
        response = requests.get(f"{API_BASE_URL}/api/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('services', data)
    
    def test_realtime_data_api(self):
        """测试实时数据API"""
        if not self.api_available:
            self.skipTest("API服务不可用")
        
        response = requests.get(f"{API_BASE_URL}/api/realtime")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        realtime_data = data['data']
        self.assertIn('timestamp', realtime_data)
        self.assertIn('total_segments', realtime_data)
        self.assertIn('segments', realtime_data)
    
    def test_prediction_api(self):
        """测试预测API"""
        if not self.api_available:
            self.skipTest("API服务不可用")
        
        test_data = {
            'incident_location': [39.9042, 116.4074],
            'prediction_time': 30,
            'impact_radius': 2
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/predict",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
    
    def test_emergency_vehicles_api(self):
        """测试应急车辆API"""
        if not self.api_available:
            self.skipTest("API服务不可用")
        
        response = requests.get(f"{API_BASE_URL}/api/emergency/vehicles")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        self.assertIsInstance(data['data'], list)
    
    def test_system_metrics_api(self):
        """测试系统指标API"""
        if not self.api_available:
            self.skipTest("API服务不可用")
        
        response = requests.get(f"{API_BASE_URL}/api/system/metrics")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('data', data)
        
        metrics = data['data']
        self.assertIn('cpu_usage', metrics)
        self.assertIn('memory_usage', metrics)
        self.assertIn('model_performance', metrics)


class DataFlowIntegrationTest(unittest.TestCase):
    """数据流集成测试"""
    
    def test_data_pipeline_flow(self):
        """测试数据管道流程"""
        if not MODELS_AVAILABLE:
            self.skipTest("模型模块不可用")
        
        # 模拟完整数据流程
        result = IntegrationTestResult()
        
        try:
            # 1. 数据预处理
            start_time = time.time()
            test_data = pd.DataFrame({
                'segment_id': [f'seg_{i}' for i in range(5)],
                'speed': np.random.uniform(30, 80, 5),
                'flow': np.random.uniform(500, 2000, 5),
                'occupancy': np.random.uniform(0.2, 0.8, 5),
                'capacity': np.random.uniform(1500, 3000, 5)
            })
            
            predictor = CongestionPropagationPredictor({'input_dim': 4})
            processed_data = predictor.prepare_data(test_data)
            duration = time.time() - start_time
            
            result.add_result("数据预处理", True, f"处理时间: {duration:.2f}秒", duration)
            
            # 2. 模型预测
            start_time = time.time()
            segments = create_sample_data(n_segments=5)
            predictions = predictor.predict_congestion_propagation(segments, prediction_horizon=3)
            duration = time.time() - start_time
            
            result.add_result("模型预测", True, f"预测时间: {duration:.2f}秒", duration)
            
            # 3. 结果验证
            if len(predictions) == 5:
                result.add_result("预测结果验证", True, "预测结果数量正确")
            else:
                result.add_result("预测结果验证", False, f"预期5个结果，实际{len(predictions)}个")
            
        except Exception as e:
            result.add_result("数据管道测试", False, f"异常: {str(e)}")
        
        # 输出测试结果摘要
        summary = result.get_summary()
        self.assertGreaterEqual(summary['success_rate'], 0.8, "数据管道测试成功率过低")
    
    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        if not MODELS_AVAILABLE:
            self.skipTest("模型模块不可用")
        
        result = IntegrationTestResult()
        
        try:
            # 模拟完整工作流程：数据→预测→调度
            start_time = time.time()
            
            # 1. 交通数据收集和预测
            predictor = CongestionPropagationPredictor({'input_dim': 4})
            segments = create_sample_data(n_segments=5)
            predictions = predictor.predict_congestion_propagation(segments, prediction_horizon=3)
            
            # 2. 瓶颈识别
            bottlenecks = predictor.identify_bottlenecks(segments)
            
            # 3. 应急调度
            dispatcher = EmergencyDispatcher("测试中心")
            from pathfinding.emergency_dispatcher import Location, EmergencyVehicle, VehicleType, EmergencyTask, TaskPriority
            
            # 添加车辆
            vehicle = EmergencyVehicle(
                vehicle_id="TEST_001",
                vehicle_type=VehicleType.MEDICAL,
                current_location=Location(0, 0, "基地")
            )
            dispatcher.add_vehicle(vehicle)
            
            # 创建应急任务
            if bottlenecks:
                task = EmergencyTask(
                    task_id="EMERGENCY_001",
                    location=Location(50, 100, "事故点"),
                    task_type=VehicleType.MEDICAL,
                    priority=TaskPriority.HIGH,
                    description="紧急救援任务"
                )
                dispatcher.add_task(task)
                dispatch_success = dispatcher.dispatch_task(task.task_id)
                
                if dispatch_success:
                    result.add_result("端到端调度", True, "应急调度成功")
                else:
                    result.add_result("端到端调度", False, "应急调度失败")
            
            duration = time.time() - start_time
            result.add_result("端到端工作流程", True, f"总耗时: {duration:.2f}秒", duration)
            
        except Exception as e:
            result.add_result("端到端工作流程", False, f"异常: {str(e)}")
        
        # 验证工作流程完整性
        summary = result.get_summary()
        self.assertGreaterEqual(summary['success_rate'], 0.8, "端到端工作流程测试失败率过高")


class PerformanceIntegrationTest(unittest.TestCase):
    """性能集成测试"""
    
    def test_model_inference_performance(self):
        """测试模型推理性能"""
        if not MODELS_AVAILABLE:
            self.skipTest("模型模块不可用")
        
        predictor = CongestionPropagationPredictor({'input_dim': 4})
        segments = create_sample_data(n_segments=10)
        
        # 多次推理测试
        inference_times = []
        for i in range(5):
            start_time = time.time()
            predictions = predictor.predict_congestion_propagation(segments, prediction_horizon=3)
            duration = time.time() - start_time
            inference_times.append(duration)
        
        avg_inference_time = np.mean(inference_times)
        max_inference_time = np.max(inference_times)
        
        # 性能验证
        self.assertLess(avg_inference_time, TestConfig.MAX_RESPONSE_TIME, "平均推理时间过长")
        self.assertLess(max_inference_time, TestConfig.MAX_RESPONSE_TIME * 2, "最大推理时间过长")
        
        print(f"平均推理时间: {avg_inference_time:.3f}秒")
        print(f"最大推理时间: {max_inference_time:.3f}秒")
    
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        if not MODELS_AVAILABLE:
            self.skipTest("模型模块不可用")
        
        import concurrent.futures
        
        predictor = CongestionPropagationPredictor({'input_dim': 4})
        segments = create_sample_data(n_segments=5)
        
        def inference_task(task_id):
            start_time = time.time()
            predictions = predictor.predict_congestion_propagation(segments, prediction_horizon=3)
            duration = time.time() - start_time
            return task_id, duration, len(predictions)
        
        # 并发执行推理任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(inference_task, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 分析结果
        durations = [result[1] for result in results]
        success_count = sum(1 for result in results if result[2] == 5)
        
        avg_duration = np.mean(durations)
        success_rate = success_count / len(results)
        
        self.assertGreaterEqual(success_rate, TestConfig.MIN_SUCCESS_RATE, "并发请求成功率过低")
        self.assertLess(avg_duration, TestConfig.MAX_RESPONSE_TIME, "并发请求平均响应时间过长")
        
        print(f"并发请求成功率: {success_rate:.2%}")
        print(f"平均响应时间: {avg_duration:.3f}秒")


def run_integration_tests():
    """运行集成测试套件"""
    print("=" * 60)
    print("智能交通流预测系统集成测试")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    if MODELS_AVAILABLE:
        test_suite.addTest(unittest.makeSuite(ModelIntegrationTest))
        test_suite.addTest(unittest.makeSuite(ServiceIntegrationTest))
        test_suite.addTest(unittest.makeSuite(PathfindingIntegrationTest))
        test_suite.addTest(unittest.makeSuite(DataFlowIntegrationTest))
        test_suite.addTest(unittest.makeSuite(PerformanceIntegrationTest))
    
    test_suite.addTest(unittest.makeSuite(APIIntegrationTest))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试摘要
    print("\n" + "=" * 60)
    print("集成测试摘要")
    print("=" * 60)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
