#!/usr/bin/env python3
"""
智能交通流预测系统 - 增强版API服务器
集成稳定性增强功能：日志系统、监控、恢复、压力测试等
"""

import os
import sys
import json
import time
import signal
import threading
import atexit
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# 添加项目路径
sys.path.append('/workspace/code')

# Flask相关导入
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging as flask_logging

# 导入原有系统模块
try:
    from models.gcn_lstm_hybrid import GCNLSTMHybrid
    from services.llm_service import LLMService
    from services.congestion_analyzer import CongestionAnalyzer
    from services.emergency_advisor import EmergencyAdvisor
    from services.prediction_explainer import PredictionExplainer
    from pathfinding.emergency_dispatcher import EmergencyDispatcher
    from pathfinding.multi_objective_planner import MultiObjectivePlanner
    
    # 导入真实数据集成模块
    from data_integration.real_data_integration import get_integration_instance
    from data_integration.config_manager import get_config_manager
    
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    print("将使用基础功能继续运行...")

# 导入稳定性增强模块
from stability.logging_system import TrafficLogger, get_logger
from stability.monitoring_system import SystemMonitor, HealthStatus
from stability.system_recovery import SystemRecoveryManager, SystemState
from stability.health_api_endpoints import register_health_api_endpoints, get_health_api_manager

# 配置日志系统
traffic_logger = TrafficLogger()
logger = get_logger('main')

# Flask应用初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic_prediction_secret_key_2025'
CORS(app, origins=["*"])
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
system_monitor = None
recovery_manager = None
health_api_manager = None
shutdown_event = threading.Event()

class EnhancedAPIServer:
    """增强版API服务器"""
    
    def __init__(self):
        self.logger = logger
        self.start_time = datetime.now()
        self.is_running = False
        
        # 初始化核心组件
        self._init_core_components()
        
        # 初始化稳定性增强功能
        self._init_stability_features()
        
        # 注册API端点
        self._register_api_endpoints()
        
        # 注册信号处理器
        self._register_signal_handlers()
        
        self.logger.info("增强版API服务器初始化完成")
    
    def _init_core_components(self):
        """初始化核心组件"""
        try:
            # 初始化数据集成
            self.data_integration = get_integration_instance()
            self.logger.info("数据集成模块初始化成功")
        except Exception as e:
            self.logger.warning(f"数据集成模块初始化失败: {str(e)}")
            self.data_integration = None
        
        try:
            # 初始化模型
            self.model = GCNLSTMHybrid()
            self.logger.info("GCN-LSTM模型初始化成功")
        except Exception as e:
            self.logger.warning(f"模型初始化失败: {str(e)}")
            self.model = None
        
        try:
            # 初始化LLM服务
            self.llm_service = LLMService()
            self.logger.info("LLM服务初始化成功")
        except Exception as e:
            self.logger.warning(f"LLM服务初始化失败: {str(e)}")
            self.llm_service = None
        
        try:
            # 初始化拥堵分析器
            self.congestion_analyzer = CongestionAnalyzer()
            self.logger.info("拥堵分析器初始化成功")
        except Exception as e:
            self.logger.warning(f"拥堵分析器初始化失败: {str(e)}")
            self.congestion_analyzer = None
        
        try:
            # 初始化应急顾问
            self.emergency_advisor = EmergencyAdvisor()
            self.logger.info("应急顾问初始化成功")
        except Exception as e:
            self.logger.warning(f"应急顾问初始化失败: {str(e)}")
            self.emergency_advisor = None
        
        try:
            # 初始化预测解释器
            self.prediction_explainer = PredictionExplainer()
            self.logger.info("预测解释器初始化成功")
        except Exception as e:
            self.logger.warning(f"预测解释器初始化失败: {str(e)}")
            self.prediction_explainer = None
        
        try:
            # 初始化应急调度器
            self.emergency_dispatcher = EmergencyDispatcher()
            self.logger.info("应急调度器初始化成功")
        except Exception as e:
            self.logger.warning(f"应急调度器初始化失败: {str(e)}")
            self.emergency_dispatcher = None
        
        try:
            # 初始化多目标规划器
            self.multi_objective_planner = MultiObjectivePlanner()
            self.logger.info("多目标规划器初始化成功")
        except Exception as e:
            self.logger.warning(f"多目标规划器初始化失败: {str(e)}")
            self.multi_objective_planner = None
    
    def _init_stability_features(self):
        """初始化稳定性增强功能"""
        try:
            # 初始化监控系统
            global system_monitor
            system_monitor = SystemMonitor()
            system_monitor.start_monitoring()
            self.logger.info("监控系统启动成功")
        except Exception as e:
            self.logger.error(f"监控系统启动失败: {str(e)}")
            system_monitor = None
        
        try:
            # 初始化恢复管理器
            global recovery_manager
            recovery_manager = SystemRecoveryManager()
            recovery_manager.state = SystemState.RUNNING
            self.logger.info("恢复管理器初始化成功")
        except Exception as e:
            self.logger.error(f"恢复管理器初始化失败: {str(e)}")
            recovery_manager = None
        
        try:
            # 初始化健康API管理器
            global health_api_manager
            health_api_manager = get_health_api_manager()
            self.logger.info("健康API管理器初始化成功")
        except Exception as e:
            self.logger.error(f"健康API管理器初始化失败: {str(e)}")
            health_api_manager = None
    
    def _register_api_endpoints(self):
        """注册API端点"""
        # 原有API端点
        self._register_original_endpoints()
        
        # 健康检查API端点
        if health_api_manager:
            register_health_api_endpoints(app)
            self.logger.info("健康检查API端点注册成功")
    
    def _register_original_endpoints(self):
        """注册原有API端点"""
        
        @app.route('/api/health', methods=['GET'])
        def health_check():
            """基础健康检查"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0-stability-enhanced',
                'uptime': str(datetime.now() - self.start_time)
            })
        
        @app.route('/api/realtime', methods=['GET'])
        def get_realtime_data():
            """获取实时交通数据"""
            try:
                start_time = time.time()
                
                # 获取实时数据
                if self.data_integration:
                    realtime_data = self.data_integration.get_realtime_data()
                else:
                    realtime_data = {'message': '数据集成模块不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('get_realtime_data', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': realtime_data,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"获取实时数据失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/predict', methods=['POST'])
        def predict_traffic():
            """交通流预测"""
            try:
                start_time = time.time()
                data = request.get_json()
                
                if not data:
                    return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
                
                # 执行预测
                if self.model:
                    # 这里应该调用实际的预测逻辑
                    prediction_result = {
                        'predicted_flow': 150.5,
                        'confidence': 0.85,
                        'prediction_time': datetime.now().isoformat()
                    }
                else:
                    prediction_result = {'message': '模型不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('predict_traffic', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': prediction_result,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"交通预测失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/emergency/vehicles', methods=['GET'])
        def get_emergency_vehicles():
            """获取应急车辆信息"""
            try:
                start_time = time.time()
                
                if self.emergency_dispatcher:
                    # 获取应急车辆位置
                    vehicles_data = self.emergency_dispatcher.get_emergency_vehicles()
                else:
                    vehicles_data = {'message': '应急调度器不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('get_emergency_vehicles', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': vehicles_data,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"获取应急车辆信息失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/weather', methods=['GET'])
        def get_weather():
            """获取天气数据"""
            try:
                start_time = time.time()
                
                if self.data_integration:
                    weather_data = self.data_integration.get_weather_data()
                else:
                    weather_data = {'message': '数据集成模块不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('get_weather', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': weather_data,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"获取天气数据失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/llm/explain', methods=['POST'])
        def explain_prediction():
            """LLM解释预测结果"""
            try:
                start_time = time.time()
                data = request.get_json()
                
                if not data:
                    return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
                
                if self.llm_service:
                    explanation = self.llm_service.explain_prediction(data)
                else:
                    explanation = {'message': 'LLM服务不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('explain_prediction', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': explanation,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"LLM解释失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/congestion/analyze', methods=['POST'])
        def analyze_congestion():
            """分析拥堵情况"""
            try:
                start_time = time.time()
                data = request.get_json()
                
                if not data:
                    return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
                
                if self.congestion_analyzer:
                    analysis = self.congestion_analyzer.analyze_congestion(data)
                else:
                    analysis = {'message': '拥堵分析器不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('analyze_congestion', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': analysis,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"拥堵分析失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/emergency/advice', methods=['POST'])
        def get_emergency_advice():
            """获取应急建议"""
            try:
                start_time = time.time()
                data = request.get_json()
                
                if not data:
                    return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
                
                if self.emergency_advisor:
                    advice = self.emergency_advisor.get_emergency_advice(data)
                else:
                    advice = {'message': '应急顾问不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('get_emergency_advice', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': advice,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"获取应急建议失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/route/optimize', methods=['POST'])
        def optimize_route():
            """优化路径"""
            try:
                start_time = time.time()
                data = request.get_json()
                
                if not data:
                    return jsonify({'status': 'error', 'message': '请求数据为空'}), 400
                
                if self.multi_objective_planner:
                    route = self.multi_objective_planner.plan_route(data)
                else:
                    route = {'message': '路径规划器不可用'}
                
                duration = time.time() - start_time
                self.logger.log_performance('optimize_route', duration)
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': route,
                    'response_time': duration
                })
                
            except Exception as e:
                self.logger.error(f"路径优化失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @app.route('/api/system/info', methods=['GET'])
        def get_system_info():
            """获取系统信息"""
            try:
                uptime = str(datetime.now() - self.start_time)
                
                system_info = {
                    'version': '3.0.0-stability-enhanced',
                    'start_time': self.start_time.isoformat(),
                    'uptime': uptime,
                    'components': {
                        'data_integration': self.data_integration is not None,
                        'model': self.model is not None,
                        'llm_service': self.llm_service is not None,
                        'congestion_analyzer': self.congestion_analyzer is not None,
                        'emergency_advisor': self.emergency_advisor is not None,
                        'prediction_explainer': self.prediction_explainer is not None,
                        'emergency_dispatcher': self.emergency_dispatcher is not None,
                        'multi_objective_planner': self.multi_objective_planner is not None,
                        'monitoring': system_monitor is not None,
                        'recovery': recovery_manager is not None,
                        'health_api': health_api_manager is not None
                    }
                }
                
                return jsonify({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': system_info
                })
                
            except Exception as e:
                self.logger.error(f"获取系统信息失败: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
    
    def _register_signal_handlers(self):
        """注册信号处理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，开始优雅停机...")
            self.graceful_shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # 注册退出处理器
        atexit.register(self._cleanup)
    
    def graceful_shutdown(self):
        """优雅停机"""
        self.logger.info("开始优雅停机...")
        self.is_running = False
        
        # 停止监控系统
        if system_monitor:
            system_monitor.stop_monitoring()
        
        # 停止恢复管理器
        if recovery_manager:
            recovery_manager.graceful_shutdown()
        
        # 设置关闭事件
        shutdown_event.set()
        
        self.logger.info("优雅停机完成")
    
    def _cleanup(self):
        """清理资源"""
        try:
            self.logger.info("清理系统资源...")
            
            # 停止所有后台任务
            shutdown_event.set()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            print(f"清理资源时发生错误: {str(e)}")
    
    def start(self, host='0.0.0.0', port=5000, debug=False):
        """启动服务器"""
        try:
            self.logger.info(f"启动增强版API服务器...")
            self.logger.info(f"服务器地址: http://{host}:{port}")
            self.logger.info(f"调试模式: {debug}")
            
            self.is_running = True
            
            # 创建初始检查点
            if recovery_manager:
                recovery_manager.create_checkpoint("服务器启动检查点")
            
            # 启动Socket.IO事件处理
            self._setup_socketio_events()
            
            # 启动服务器
            self.logger.info("服务器启动成功，开始接受请求...")
            socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)
            
        except Exception as e:
            self.logger.error(f"服务器启动失败: {str(e)}")
            raise
    
    def _setup_socketio_events(self):
        """设置Socket.IO事件"""
        
        @socketio.on('connect')
        def handle_connect():
            self.logger.info('客户端已连接')
            emit('status', {'message': '已连接到智能交通流预测系统'})
        
        @socketio.on('disconnect')
        def handle_disconnect():
            self.logger.info('客户端已断开连接')
        
        @socketio.on('request_realtime_data')
        def handle_realtime_data_request():
            try:
                if self.data_integration:
                    data = self.data_integration.get_realtime_data()
                    emit('realtime_data', data)
                else:
                    emit('error', {'message': '数据集成模块不可用'})
            except Exception as e:
                self.logger.error(f"实时数据推送失败: {str(e)}")
                emit('error', {'message': str(e)})
        
        @socketio.on('request_health_status')
        def handle_health_status_request():
            try:
                if health_api_manager:
                    health_status = health_api_manager.get_system_health()
                    emit('health_status', health_status)
                else:
                    emit('error', {'message': '健康检查模块不可用'})
            except Exception as e:
                self.logger.error(f"健康状态推送失败: {str(e)}")
                emit('error', {'message': str(e)})

# 创建全局服务器实例
enhanced_server = EnhancedAPIServer()

def create_app():
    """创建Flask应用"""
    return app

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='智能交通流预测系统 - 增强版API服务器')
    parser.add_argument('--host', default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 启动服务器
    enhanced_server.start(host=args.host, port=args.port, debug=args.debug)

if __name__ == '__main__':
    main()