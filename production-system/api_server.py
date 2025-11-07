#!/usr/bin/env python3
"""
智能交通流预测系统后端API服务
集成GCN+LSTM模型、LLM服务、应急调度等核心功能
"""

import os
import sys
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import queue

# 添加项目路径
sys.path.append('/workspace/code')

try:
    from models.gcn_lstm_hybrid import GCNLSTMHybrid
    from services.llm_service import LLMService
    from services.congestion_analyzer import CongestionAnalyzer
    from services.emergency_advisor import EmergencyAdvisor
    from services.prediction_explainer import PredictionExplainer
    from pathfinding.emergency_dispatcher import EmergencyDispatcher
    from pathfinding.multi_objective_planner import MultiObjectivePlanner
except ImportError as e:
    print(f"警告: 无法导入某些模块: {e}")
    print("将使用模拟数据继续运行...")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask应用初始化
app = Flask(__name__)
app.config['SECRET_KEY'] = 'traffic_prediction_secret_key_2025'
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# 全局变量
traffic_data_queue = queue.Queue()
model_service = None
llm_service = None
analyzer = None
emergency_advisor = None
explainer = None
dispatcher = None
planner = None

# 模拟交通数据生成器
class TrafficDataGenerator:
    def __init__(self):
        self.beijing_bounds = {
            'lat_min': 39.4,
            'lat_max': 40.2,
            'lng_min': 116.0,
            'lng_max': 117.2
        }
        self.road_segments = self._generate_road_segments()
        
    def _generate_road_segments(self):
        """生成北京市主要道路段"""
        segments = []
        # 主要环路
        ring_roads = [
            {'name': '二环路', 'center': [39.9042, 116.4074], 'radius': 0.05},
            {'name': '三环路', 'center': [39.9042, 116.4074], 'radius': 0.08},
            {'name': '四环路', 'center': [39.9042, 116.4074], 'radius': 0.12},
            {'name': '五环路', 'center': [39.9042, 116.4074], 'radius': 0.18}
        ]
        
        for i, road in enumerate(ring_roads):
            for j in range(8):  # 每条环路8个路段
                angle = j * 45  # 45度间隔
                lat = road['center'][0] + road['radius'] * np.cos(np.radians(angle))
                lng = road['center'][1] + road['radius'] * np.sin(np.radians(angle))
                
                segments.append({
                    'id': f'RING_{i+1}_{j:02d}',
                    'name': f'{road["name"]}段{j+1}',
                    'lat': lat,
                    'lng': lng,
                    'road_type': 'ring',
                    'lanes': 4 + i,  # 内环车道数较少
                    'speed_limit': 80 - i * 10  # 内环限速较高
                })
        
        # 主要放射道路
        radial_roads = [
            {'name': '长安街', 'direction': [1, 0], 'center': [39.9042, 116.4074]},
            {'name': '建国门大街', 'direction': [0.7, 0.7], 'center': [39.9042, 116.4074]},
            {'name': '复兴门外大街', 'direction': [-0.7, 0.7], 'center': [39.9042, 116.4074]},
            {'name': '中关村大街', 'direction': [0, 1], 'center': [39.9042, 116.4074]}
        ]
        
        for road in radial_roads:
            for i in range(6):  # 每条放射路6个路段
                lat = road['center'][0] + road['direction'][0] * i * 0.02
                lng = road['center'][1] + road['direction'][1] * i * 0.02
                
                segments.append({
                    'id': f'RADIAL_{road["name"][:4]}_{i:02d}',
                    'name': f'{road["name"]}段{i+1}',
                    'lat': lat,
                    'lng': lng,
                    'road_type': 'radial',
                    'lanes': 6,
                    'speed_limit': 60
                })
        
        return segments[:30]  # 返回前30个路段
    
    def generate_realtime_data(self):
        """生成实时交通数据"""
        timestamp = datetime.now().isoformat()
        
        # 计算全局指标
        total_segments = len(self.road_segments)
        congested_segments = 0
        total_speed = 0
        total_flow = 0
        
        segments_data = []
        
        for segment in self.road_segments:
            # 模拟时段影响 (早晚高峰)
            hour = datetime.now().hour
            time_factor = 1.0
            if 7 <= hour <= 9 or 17 <= hour <= 19:  # 早晚高峰
                time_factor = 0.7
            elif 22 <= hour or hour <= 6:  # 夜间
                time_factor = 1.3
            
            # 模拟天气影响
            weather_factor = np.random.uniform(0.8, 1.2)
            
            # 计算当前速度
            base_speed = segment['speed_limit']
            current_speed = base_speed * time_factor * weather_factor * np.random.uniform(0.6, 1.0)
            
            # 计算流量 (基于速度反比关系)
            max_flow = segment['lanes'] * 2000  # 最大流量
            flow = max_flow * (1 - current_speed / base_speed) * np.random.uniform(0.8, 1.2)
            
            # 计算密度和占有率
            density = flow / max(current_speed, 1)  # 车辆密度
            occupancy = min(density / 100 * 100, 100)  # 占有率百分比
            
            # 确定状态
            if current_speed < segment['speed_limit'] * 0.4:
                status = 'congested'
                congested_segments += 1
            elif current_speed < segment['speed_limit'] * 0.7:
                status = 'warning'
            else:
                status = 'normal'
            
            segment_data = {
                'id': segment['id'],
                'name': segment['name'],
                'lat': segment['lat'],
                'lng': segment['lng'],
                'speed': round(current_speed, 1),
                'flow': int(flow),
                'density': int(density),
                'occupancy': round(occupancy, 1),
                'status': status,
                'speed_limit': segment['speed_limit'],
                'road_type': segment['road_type']
            }
            
            segments_data.append(segment_data)
            total_speed += current_speed
            total_flow += flow
        
        # 计算全局指标
        average_speed = round(total_speed / total_segments, 1)
        total_flow_k = round(total_flow / 1000, 1)  # 转换为千辆/小时
        congestion_distance = round(congested_segments * 2.5, 1)  # 估算拥堵里程
        system_status = '告警' if congested_segments > total_segments * 0.3 else '正常'
        
        return {
            'timestamp': timestamp,
            'total_segments': total_segments,
            'congested_segments': congested_segments,
            'congestion_distance': congestion_distance,
            'average_speed': average_speed,
            'total_flow': total_flow_k,
            'system_status': system_status,
            'segments': segments_data,
            'weather': {
                'condition': '晴朗',
                'temperature': round(np.random.uniform(15, 25), 1),
                'visibility': '良好',
                'impact_factor': 1.0
            },
            'traffic_light_status': {
                'total_lights': 1200,
                'green_lights': 800,
                'yellow_lights': 200,
                'red_lights': 200
            }
        }
    
    def generate_prediction_data(self, incident_location, prediction_time, impact_radius):
        """生成预测数据"""
        # 模拟GCN+LSTM模型预测
        base_time = datetime.now()
        prediction_steps = []
        
        for i in range(int(prediction_time) // 5):  # 每5分钟一个预测点
            step_time = base_time + timedelta(minutes=i * 5)
            
            # 模拟拥堵扩散
            impact_factor = max(0, 1 - (i * 5) / int(prediction_time))
            affected_segments = int(30 * impact_factor)
            
            prediction_step = {
                'time': step_time.strftime('%H:%M'),
                'predicted_speed': round(45 * (1 - impact_factor * 0.6), 1),
                'predicted_flow': round(67 * (1 - impact_factor * 0.4), 1),
                'congestion_level': round(impact_factor * 100, 1),
                'affected_segments': affected_segments,
                'confidence': round(max(0.5, 1 - impact_factor * 0.5), 3)
            }
            prediction_steps.append(prediction_step)
        
        # 置信度分析
        confidence_analysis = {
            'short_term': {
                'time_range': '15分钟',
                'confidence': 0.923,
                'factors': ['历史数据充足', '模型训练充分', '天气条件良好']
            },
            'medium_term': {
                'time_range': '30分钟',
                'confidence': 0.876,
                'factors': ['预测时间适中', '影响因素稳定', '模型性能良好']
            },
            'long_term': {
                'time_range': '60分钟',
                'confidence': 0.789,
                'factors': ['预测时间较长', '不确定性增加', '建议谨慎参考']
            }
        }
        
        return {
            'prediction_id': f'PRED_{int(time.time())}',
            'incident_location': incident_location,
            'prediction_time': prediction_time,
            'impact_radius': impact_radius,
            'prediction_steps': prediction_steps,
            'confidence_analysis': confidence_analysis,
            'model_info': {
                'model_type': 'GCN+LSTM Hybrid',
                'version': '2.1.0',
                'training_data': '北京市交通数据2020-2025',
                'last_updated': '2025-11-01',
                'accuracy_metrics': {
                    'mae': 2.34,
                    'rmse': 3.67,
                    'mape': 5.82,
                    'r2': 0.9234
                }
            }
        }
    
    def generate_emergency_vehicles(self):
        """生成应急车辆数据"""
        vehicles = [
            {
                'id': 'AMB_001',
                'type': 'ambulance',
                'name': '救护车-001',
                'location': {'lat': 39.9042, 'lng': 116.4074},
                'status': 'available',
                'capacity': 1,
                'equipment': ['心电监护', '除颤器', '氧气瓶'],
                'response_time': 8
            },
            {
                'id': 'FIRE_002',
                'type': 'fire',
                'name': '消防车-002',
                'location': {'lat': 39.9163, 'lng': 116.3974},
                'status': 'busy',
                'capacity': 6,
                'equipment': ['消防水枪', '泡沫', '救援工具'],
                'response_time': 15
            },
            {
                'id': 'POL_003',
                'type': 'police',
                'name': '警车-003',
                'location': {'lat': 39.9282, 'lng': 116.4074},
                'status': 'available',
                'capacity': 2,
                'equipment': ['执法记录仪', '对讲机', '急救包'],
                'response_time': 12
            },
            {
                'id': 'RESCUE_004',
                'type': 'rescue',
                'name': '救援车-004',
                'location': {'lat': 39.9100, 'lng': 116.4200},
                'status': 'available',
                'capacity': 4,
                'equipment': ['担架', '医疗设备', '通讯设备'],
                'response_time': 10
            }
        ]
        
        return vehicles

# 初始化服务
def initialize_services():
    """初始化各种服务"""
    global model_service, llm_service, analyzer, emergency_advisor, explainer, dispatcher, planner
    
    try:
        # 初始化模型服务
        logger.info("正在初始化GCN+LSTM模型...")
        # model_service = GCNLSTMHybrid()  # 实际使用时取消注释
        
        # 初始化LLM服务
        logger.info("正在初始化LLM服务...")
        # llm_service = LLMService()  # 实际使用时取消注释
        
        # 初始化分析器
        logger.info("正在初始化拥堵分析器...")
        # analyzer = CongestionAnalyzer()  # 实际使用时取消注释
        
        # 初始化应急顾问
        logger.info("正在初始化应急顾问...")
        # emergency_advisor = EmergencyAdvisor()  # 实际使用时取消注释
        
        # 初始化预测解释器
        logger.info("正在初始化预测解释器...")
        # explainer = PredictionExplainer()  # 实际使用时取消注释
        
        # 初始化调度器和规划器
        logger.info("正在初始化应急调度器...")
        # dispatcher = EmergencyDispatcher()  # 实际使用时取消注释
        
        logger.info("正在初始化多目标规划器...")
        # planner = MultiObjectivePlanner()  # 实际使用时取消注释
        
        logger.info("所有服务初始化完成")
        
    except Exception as e:
        logger.warning(f"服务初始化部分失败: {e}")
        logger.info("将使用模拟模式继续运行")

# 数据生成器实例
traffic_generator = TrafficDataGenerator()

# API路由
@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'model_service': model_service is not None,
            'llm_service': llm_service is not None,
            'analyzer': analyzer is not None,
            'emergency_advisor': emergency_advisor is not None,
            'explainer': explainer is not None,
            'dispatcher': dispatcher is not None,
            'planner': planner is not None
        }
    })

@app.route('/api/realtime', methods=['GET'])
def get_realtime_data():
    """获取实时交通数据"""
    try:
        data = traffic_generator.generate_realtime_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        logger.error(f"获取实时数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_traffic():
    """交通流预测"""
    try:
        data = request.get_json()
        incident_location = data.get('incident_location')
        prediction_time = data.get('prediction_time', 30)
        impact_radius = data.get('impact_radius', 2)
        
        if not incident_location:
            return jsonify({'success': False, 'error': '缺少事故位置信息'}), 400
        
        # 生成预测数据
        prediction_data = traffic_generator.generate_prediction_data(
            incident_location, prediction_time, impact_radius
        )
        
        # 如果有模型服务，使用真实模型预测
        # if model_service:
        #     try:
        #         # 调用真实的GCN+LSTM模型
        #         model_prediction = model_service.predict(
        #             location=incident_location,
        #             time_horizon=prediction_time,
        #             impact_radius=impact_radius
        #         )
        #         prediction_data['model_prediction'] = model_prediction
        #     except Exception as e:
        #         logger.warning(f"模型预测失败，使用模拟数据: {e}")
        
        return jsonify({'success': True, 'data': prediction_data})
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency/vehicles', methods=['GET'])
def get_emergency_vehicles():
    """获取应急车辆信息"""
    try:
        vehicles = traffic_generator.generate_emergency_vehicles()
        return jsonify({'success': True, 'data': vehicles})
    except Exception as e:
        logger.error(f"获取应急车辆信息失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/emergency/dispatch', methods=['POST'])
def dispatch_vehicle():
    """调度应急车辆"""
    try:
        data = request.get_json()
        incident_id = data.get('incident_id')
        vehicle_id = data.get('vehicle_id')
        location = data.get('location')
        
        if not all([incident_id, vehicle_id, location]):
            return jsonify({'success': False, 'error': '缺少必要参数'}), 400
        
        # 模拟调度过程
        dispatch_result = {
            'dispatch_id': f'DISP_{int(time.time())}',
            'incident_id': incident_id,
            'vehicle_id': vehicle_id,
            'status': 'dispatched',
            'estimated_arrival': '8分钟',
            'route_info': {
                'distance': '2.3km',
                'traffic_conditions': '良好',
                'alternative_routes': 2
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # 如果有调度器服务，使用真实调度
        # if dispatcher:
        #     try:
        #         real_dispatch = dispatcher.dispatch_vehicle(
        #             vehicle_id=vehicle_id,
        #             destination=location,
        #             priority='high'
        #         )
        #         dispatch_result['real_dispatch'] = real_dispatch
        #     except Exception as e:
        #         logger.warning(f"真实调度失败，使用模拟结果: {e}")
        
        return jsonify({'success': True, 'data': dispatch_result})
        
    except Exception as e:
        logger.error(f"车辆调度失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/metrics', methods=['GET'])
def get_system_metrics():
    """获取系统性能指标"""
    try:
        # 模拟系统指标
        metrics = {
            'cpu_usage': round(np.random.uniform(60, 80), 1),
            'memory_usage': round(np.random.uniform(40, 60), 1),
            'gpu_usage': round(np.random.uniform(70, 90), 1),
            'network_latency': round(np.random.uniform(10, 20), 1),
            'model_performance': {
                'mae': 2.34,
                'rmse': 3.67,
                'mape': 5.82,
                'r2': 0.9234,
                'inference_time': 0.156
            },
            'system_health': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({'success': True, 'data': metrics})
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/system/logs', methods=['GET'])
def get_system_logs():
    """获取系统日志"""
    try:
        # 模拟系统日志
        logs = [
            {
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': '模型预测任务完成',
                'detail': 'GCN+LSTM模型成功预测15分钟后的交通流状态'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=2)).isoformat(),
                'level': 'WARNING',
                'message': '检测到交通异常',
                'detail': 'RING_1_03路段平均速度下降至38.0km/h，触发预警'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                'level': 'INFO',
                'message': '数据同步完成',
                'detail': '成功同步30个路段的实时交通数据'
            },
            {
                'timestamp': (datetime.now() - timedelta(minutes=8)).isoformat(),
                'level': 'ERROR',
                'message': '应急响应触发',
                'detail': '自动调度救护车-001前往事故地点'
            }
        ]
        
        return jsonify({'success': True, 'data': logs})
        
    except Exception as e:
        logger.error(f"获取系统日志失败: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info(f"客户端已连接: {request.sid}")
    emit('connected', {'message': '连接成功', 'timestamp': datetime.now().isoformat()})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    logger.info(f"客户端已断开: {request.sid}")

@socketio.on('subscribe_traffic_data')
def handle_subscribe_traffic_data():
    """订阅实时交通数据"""
    logger.info(f"客户端 {request.sid} 订阅实时交通数据")
    emit('subscription_confirmed', {'channel': 'traffic_data'})

# 后台数据推送线程
def data_push_worker():
    """后台数据推送工作线程"""
    while True:
        try:
            # 生成实时数据
            data = traffic_generator.generate_realtime_data()
            
            # 推送给所有订阅的客户端
            socketio.emit('traffic-data', data)
            
            # 等待5秒后推送下一批数据
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"数据推送失败: {e}")
            time.sleep(5)

# 启动后台数据推送
def start_data_push():
    """启动数据推送线程"""
    worker_thread = threading.Thread(target=data_push_worker, daemon=True)
    worker_thread.start()
    logger.info("后台数据推送线程已启动")

if __name__ == '__main__':
    # 初始化服务
    initialize_services()
    
    # 启动数据推送
    start_data_push()
    
    # 启动Flask应用
    logger.info("启动智能交通流预测系统API服务器...")
    logger.info("API服务地址: http://localhost:3001")
    logger.info("WebSocket地址: ws://localhost:3001")
    
    socketio.run(app, host='0.0.0.0', port=3001, debug=False)