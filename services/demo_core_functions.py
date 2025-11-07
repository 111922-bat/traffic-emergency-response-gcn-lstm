"""
拥堵原因分析器简化示例

专注于核心分析功能，不依赖复杂的深度学习模型

作者：TrafficAI Team
日期：2025-11-05
"""

import numpy as np
import pandas as pd
import time
import json
from datetime import datetime, timedelta

from congestion_analyzer import (
    WeatherAnalyzer,
    IncidentDetector,
    EventAnalyzer,
    CausalInferenceEngine,
    PropagationAnalyzer,
    TimeSeriesAnalyzer,
    RiskAssessmentEngine,
    CauseType,
    RiskLevel,
    PropagationDirection,
    CongestionCause,
    CausalRelationship,
    PropagationPath,
    RiskAssessment
)


def demo_weather_analysis():
    """天气影响分析演示"""
    print("=" * 60)
    print("天气影响分析演示")
    print("=" * 60)
    
    weather_analyzer = WeatherAnalyzer()
    
    # 模拟不同天气条件
    weather_scenarios = [
        {
            "name": "晴朗天气",
            "data": {
                'rain_intensity': 0,
                'snow_intensity': 0,
                'visibility': 2000,
                'wind_speed': 5.0,
                'temperature': 22.0
            }
        },
        {
            "name": "小雨天气",
            "data": {
                'rain_intensity': 10.0,
                'snow_intensity': 0,
                'visibility': 1500,
                'wind_speed': 8.0,
                'temperature': 15.0
            }
        },
        {
            "name": "暴雨天气",
            "data": {
                'rain_intensity': 45.0,
                'snow_intensity': 0,
                'visibility': 400,
                'wind_speed': 18.0,
                'temperature': 8.0
            }
        },
        {
            "name": "雪天天气",
            "data": {
                'rain_intensity': 0,
                'snow_intensity': 8.0,
                'visibility': 300,
                'wind_speed': 12.0,
                'temperature': -3.0
            }
        }
    ]
    
    # 模拟交通数据
    traffic_data = pd.DataFrame({
        'segment_id': ['highway_001', 'bridge_001', 'city_road_001'],
        'speed': [65, 45, 35],
        'flow': [1800, 1200, 800],
        'occupancy': [0.4, 0.7, 0.8]
    })
    
    print("分析不同天气条件对交通的影响:")
    
    for scenario in weather_scenarios:
        impact_score, impact_factors = weather_analyzer.analyze_weather_impact(
            scenario['data'], traffic_data
        )
        
        print(f"\n  {scenario['name']}:")
        print(f"    影响评分: {impact_score:.2f}")
        print(f"    影响因子: {len(impact_factors)}个")
        if impact_factors:
            print(f"    主要因子: {', '.join(impact_factors[:3])}")
        
        # 根据影响评分给出建议
        if impact_score > 0.7:
            print(f"    建议: ⚠️  严重天气影响，建议采取紧急措施")
        elif impact_score > 0.4:
            print(f"    建议: ⚡ 中等天气影响，建议加强监控")
        else:
            print(f"    建议: ✓ 天气影响较小，常规监控即可")


def demo_incident_detection():
    """事故检测演示"""
    print("\n" + "=" * 60)
    print("交通事故检测演示")
    print("=" * 60)
    
    incident_detector = IncidentDetector()
    
    # 模拟当前交通异常数据
    current_data = pd.DataFrame({
        'segment_id': ['highway_001', 'highway_002', 'highway_003', 'bridge_001'],
        'speed': [25, 58, 62, 15],  # highway_001和bridge_001速度异常
        'flow': [800, 1750, 1800, 600],  # 对应流量也异常
        'occupancy': [0.9, 0.45, 0.42, 0.95]  # 占有率很高
    })
    
    # 模拟历史正常数据
    historical_data = pd.DataFrame({
        'segment_id': ['highway_001'] * 20 + ['bridge_001'] * 20 + ['highway_002'] * 20 + ['highway_003'] * 20,
        'timestamp': [time.time() - i * 300 for i in range(20) for _ in range(4)],
        'speed': [60, 62, 58, 61] * 20,  # 正常速度范围
        'flow': [1700, 1650, 1750, 1680] * 20,  # 正常流量范围
        'occupancy': [0.4, 0.42, 0.38, 0.41] * 20  # 正常占有率
    })
    
    print("检测交通事故...")
    incidents = incident_detector.detect_incidents(current_data, historical_data)
    
    print(f"检测结果: 发现 {len(incidents)} 个疑似交通事故")
    
    for i, incident in enumerate(incidents, 1):
        print(f"\n  事故 {i}:")
        print(f"    位置: {incident.location}")
        print(f"    严重程度: {incident.severity:.2f}")
        print(f"    影响评分: {incident.impact_score:.2f}")
        print(f"    可信度: {incident.confidence:.2f}")
        print(f"    描述: {incident.description}")
        print(f"    检测方法: {incident.metadata.get('detection_method', 'unknown')}")


def demo_event_analysis():
    """事件影响分析演示"""
    print("\n" + "=" * 60)
    print("特殊事件影响分析演示")
    print("=" * 60)
    
    event_analyzer = EventAnalyzer()
    
    # 模拟大型活动数据
    traffic_data = pd.DataFrame({
        'segment_id': ['stadium_road_1', 'stadium_road_2', 'stadium_road_3', 'parking_area_1'],
        'longitude': [116.397, 116.398, 116.396, 116.399],
        'latitude': [39.908, 39.909, 39.907, 39.910]
    })
    
    # 创建当前时间的事件
    now = pd.Timestamp.now()
    calendar_data = pd.DataFrame({
        'id': ['football_match_001', 'concert_001', 'conference_001'],
        'name': ['足球比赛', '大型音乐会', '科技会议'],
        'type': ['sports_event', 'concert', 'conference'],
        'start_time': [
            now - pd.Timedelta(minutes=30),  # 30分钟前开始
            now + pd.Timedelta(minutes=15),  # 15分钟后开始
            now + pd.Timedelta(hours=2)      # 2小时后开始
        ],
        'end_time': [
            now + pd.Timedelta(hours=2),     # 2小时后结束
            now + pd.Timedelta(hours=3),     # 3小时后结束
            now + pd.Timedelta(hours=8)      # 8小时后结束
        ],
        'attendance': [50000, 8000, 2000],
        'longitude': [116.397, 116.398, 116.399],
        'latitude': [39.908, 39.909, 39.910]
    })
    
    print("检测特殊事件...")
    events = event_analyzer.detect_events(traffic_data, calendar_data)
    
    print(f"检测结果: 发现 {len(events)} 个影响交通的特殊事件")
    
    for i, event in enumerate(events, 1):
        print(f"\n  事件 {i}:")
        print(f"    类型: {event.cause_type.value}")
        print(f"    严重程度: {event.severity:.2f}")
        print(f"    影响评分: {event.impact_score:.2f}")
        print(f"    位置: {event.location}")
        print(f"    描述: {event.description}")
        print(f"    受影响路段: {len(event.affected_segments)}个")
        print(f"    预期观众: {event.metadata.get('expected_attendance', 0):,}人")


def demo_causal_inference():
    """因果关系推理演示"""
    print("\n" + "=" * 60)
    print("因果关系推理演示")
    print("=" * 60)
    
    causal_engine = CausalInferenceEngine()
    
    # 模拟交通数据
    traffic_data = pd.DataFrame({
        'segment_id': ['highway_001', 'highway_002', 'bridge_001', 'city_001'],
        'speed': [30, 45, 25, 50],
        'flow': [1600, 1400, 800, 1200],
        'occupancy': [0.8, 0.6, 0.9, 0.5],
        'v_c_ratio': [0.8, 0.7, 0.4, 0.6]
    })
    
    # 模拟拥堵原因
    causes = [
        CongestionCause(
            cause_id='accident_001',
            cause_type=CauseType.ACCIDENT,
            location=(116.397, 39.908),
            severity=0.9,
            start_time=time.time(),
            affected_segments=['highway_001']
        ),
        CongestionCause(
            cause_id='weather_001',
            cause_type=CauseType.WEATHER,
            location=(116.398, 39.909),
            severity=0.6,
            start_time=time.time(),
            affected_segments=['bridge_001']
        )
    ]
    
    print("分析因果关系...")
    relationships = causal_engine.build_causal_graph(traffic_data, causes)
    
    print(f"发现 {len(relationships)} 个因果关系:")
    
    for i, rel in enumerate(relationships, 1):
        print(f"\n  因果关系 {i}:")
        print(f"    原因: {rel.cause_id}")
        print(f"    结果: {rel.effect_id}")
        print(f"    因果强度: {rel.causal_strength:.2f}")
        print(f"    时间延迟: {rel.time_lag:.0f}分钟")
        print(f"    可信度: {rel.confidence:.2f}")
        print(f"    关系类型: {rel.relationship_type}")


def demo_propagation_analysis():
    """传播路径分析演示"""
    print("\n" + "=" * 60)
    print("拥堵传播路径分析演示")
    print("=" * 60)
    
    propagation_analyzer = PropagationAnalyzer()
    
    # 模拟拥堵原因
    causes = [
        CongestionCause(
            cause_id='accident_highway',
            cause_type=CauseType.ACCIDENT,
            location=(116.397, 39.908),
            severity=0.8,
            start_time=time.time(),
            affected_segments=['highway_001']
        ),
        CongestionCause(
            cause_id='event_downtown',
            cause_type=CauseType.EVENT,
            location=(116.398, 39.909),
            severity=0.7,
            start_time=time.time(),
            affected_segments=['city_center_001']
        )
    ]
    
    # 模拟交通网络
    traffic_network = {
        'segments': [
            'highway_001', 'highway_002', 'highway_003',
            'bridge_001', 'bridge_002',
            'city_center_001', 'city_center_002', 'city_center_003'
        ],
        'connections': {
            'highway_001': ['highway_002', 'bridge_001'],
            'highway_002': ['highway_001', 'highway_003'],
            'highway_003': ['highway_002'],
            'bridge_001': ['highway_001', 'city_center_001'],
            'bridge_002': ['city_center_002'],
            'city_center_001': ['bridge_001', 'city_center_002', 'city_center_003'],
            'city_center_002': ['city_center_001', 'bridge_002'],
            'city_center_003': ['city_center_001']
        }
    }
    
    print("分析拥堵传播路径...")
    propagation_paths = propagation_analyzer.analyze_propagation_paths(causes, traffic_network)
    
    print(f"发现 {len(propagation_paths)} 个传播路径:")
    
    for i, path in enumerate(propagation_paths, 1):
        print(f"\n  传播路径 {i}:")
        print(f"    源头: {path.source_segment}")
        print(f"    目标: {', '.join(path.target_segments)}")
        print(f"    方向: {path.direction.value}")
        print(f"    传播速度: {path.propagation_speed:.1f} km/h")
        print(f"    影响范围: {path.influence_range:.1f} km")
        print(f"    衰减因子: {path.attenuation_factor:.2f}")
        print(f"    路径节点: {' -> '.join(path.path_nodes)}")


def demo_time_series_analysis():
    """时间序列分析演示"""
    print("\n" + "=" * 60)
    print("时间序列分析演示")
    print("=" * 60)
    
    time_series_analyzer = TimeSeriesAnalyzer()
    
    # 模拟一周的交通数据
    dates = pd.date_range(start='2025-01-01', periods=168, freq='h')  # 一周，168小时
    
    # 模拟多个路段的时间序列数据
    segments = ['highway_001', 'bridge_001', 'city_road_001']
    time_series_data = []
    
    for segment in segments:
        for i, date in enumerate(dates):
            hour = date.hour
            day_of_week = date.dayofweek
            
            # 模拟交通模式：工作日早晚高峰
            if day_of_week < 5:  # 工作日
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰期
                    base_speed = 30 if segment == 'highway_001' else 25 if segment == 'bridge_001' else 20
                    base_flow = 2000 if segment == 'highway_001' else 1500 if segment == 'bridge_001' else 1000
                else:
                    base_speed = 60 if segment == 'highway_001' else 45 if segment == 'bridge_001' else 35
                    base_flow = 1200 if segment == 'highway_001' else 900 if segment == 'bridge_001' else 600
            else:  # 周末
                base_speed = 55 if segment == 'highway_001' else 40 if segment == 'bridge_001' else 30
                base_flow = 1000 if segment == 'highway_001' else 800 if segment == 'bridge_001' else 500
            
            # 添加随机变化
            speed = base_speed + np.random.normal(0, 5)
            flow = base_flow + np.random.normal(0, 100)
            occupancy = max(0.1, min(0.9, (flow / (base_flow + 100)) * 0.5 + np.random.normal(0, 0.1)))
            
            time_series_data.append({
                'segment_id': segment,
                'timestamp': date.timestamp(),
                'speed': speed,
                'flow': flow,
                'occupancy': occupancy
            })
    
    time_series_df = pd.DataFrame(time_series_data)
    
    print("分析时间序列趋势...")
    analysis_result = time_series_analyzer.analyze_trends(time_series_df, [])
    
    print("各路段趋势分析结果:")
    
    for segment in segments:
        if segment in analysis_result['trend_direction']:
            trend_info = analysis_result['trend_direction'][segment]
            strength_info = analysis_result['trend_strength'][segment]
            
            print(f"\n  {segment}:")
            print(f"    趋势方向: {trend_info}")
            print(f"    趋势强度: {strength_info:.2f}")
            
            if segment in analysis_result['seasonal_patterns']:
                seasonal = analysis_result['seasonal_patterns'][segment]
                print(f"    高峰时段: {seasonal['peak_hours']}")
                print(f"    工作日平均速度: {seasonal['weekday_avg_speed']:.1f} km/h")
                print(f"    周末平均速度: {seasonal['weekend_avg_speed']:.1f} km/h")
                print(f"    高峰/平峰比: {seasonal['peak_to_offpeak_ratio']:.2f}")


def demo_risk_assessment():
    """风险评估演示"""
    print("\n" + "=" * 60)
    print("综合风险评估演示")
    print("=" * 60)
    
    risk_engine = RiskAssessmentEngine()
    
    # 模拟复杂的拥堵情况
    causes = [
        CongestionCause(
            cause_id='accident_major',
            cause_type=CauseType.ACCIDENT,
            location=(116.397, 39.908),
            severity=0.9,
            start_time=time.time(),
            affected_segments=['highway_001', 'highway_002'],
            impact_score=0.8,
            confidence=0.9,
            description="高速公路重大交通事故"
        ),
        CongestionCause(
            cause_id='weather_heavy_rain',
            cause_type=CauseType.WEATHER,
            location=(116.398, 39.909),
            severity=0.7,
            start_time=time.time(),
            affected_segments=['bridge_001', 'bridge_002'],
            impact_score=0.6,
            confidence=0.8,
            description="持续强降雨天气"
        ),
        CongestionCause(
            cause_id='event_concert',
            cause_type=CauseType.EVENT,
            location=(116.399, 39.910),
            severity=0.6,
            start_time=time.time(),
            affected_segments=['city_center_001'],
            impact_score=0.5,
            confidence=0.7,
            description="大型音乐会活动"
        )
    ]
    
    propagation_paths = [
        PropagationPath(
            source_segment='highway_001',
            target_segments=['highway_002', 'bridge_001'],
            direction=PropagationDirection.DOWNSTREAM,
            propagation_speed=12.0,
            influence_range=8.0,
            attenuation_factor=0.7
        ),
        PropagationPath(
            source_segment='bridge_001',
            target_segments=['city_center_001', 'city_center_002'],
            direction=PropagationDirection.BIDIRECTIONAL,
            propagation_speed=8.0,
            influence_range=5.0,
            attenuation_factor=0.6
        )
    ]
    
    time_series_analysis = {
        'trend_direction': {
            'highway_001': 'deteriorating',
            'bridge_001': 'deteriorating',
            'city_center_001': 'stable'
        },
        'trend_strength': {
            'highway_001': 0.8,
            'bridge_001': 0.7,
            'city_center_001': 0.3
        }
    }
    
    print("执行综合风险评估...")
    risk_assessment = risk_engine.assess_risk(causes, propagation_paths, time_series_analysis)
    
    print(f"\n综合风险评估结果:")
    print(f"  风险评分: {risk_assessment.risk_score:.2f}")
    print(f"  风险等级: {risk_assessment.overall_risk_level.name}")
    print(f"  预警等级: {risk_assessment.alert_level}")
    
    print(f"\n风险因子分析:")
    for factor, score in risk_assessment.risk_factors.items():
        print(f"  {factor}: {score:.2f}")
    
    print(f"\n预测信息:")
    for key, value in risk_assessment.predictions.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n建议措施:")
    for i, rec in enumerate(risk_assessment.recommendations[:5], 1):
        print(f"  {i}. {rec}")


def demo_comprehensive_scenario():
    """综合场景演示"""
    print("\n" + "=" * 60)
    print("综合交通拥堵场景分析")
    print("=" * 60)
    
    print("场景描述:")
    print("  时间: 工作日晚高峰 (18:30)")
    print("  地点: 北京市朝阳区某主要交通枢纽")
    print("  天气: 中到大雨，能见度较低")
    print("  事件: 附近体育馆有大型体育比赛")
    
    # 模拟当前交通状况
    current_traffic = {
        'main_highway': {
            'length': 2.5, 'lanes': 4, 'capacity': 4000,
            'free_flow_speed': 80, 'current_speed': 25,
            'current_flow': 3800, 'occupancy': 0.85
        },
        'connecting_bridge': {
            'length': 1.2, 'lanes': 2, 'capacity': 2000,
            'free_flow_speed': 60, 'current_speed': 20,
            'current_flow': 1900, 'occupancy': 0.9
        },
        'city_arterial': {
            'length': 1.8, 'lanes': 3, 'capacity': 3000,
            'free_flow_speed': 50, 'current_speed': 30,
            'current_flow': 2800, 'occupancy': 0.75
        },
        'stadium_area': {
            'length': 1.0, 'lanes': 2, 'capacity': 1800,
            'free_flow_speed': 40, 'current_speed': 15,
            'current_flow': 1700, 'occupancy': 0.95
        }
    }
    
    # 天气数据
    weather_data = {
        'rain_intensity': 35.0,
        'visibility': 600,
        'wind_speed': 15.0,
        'temperature': 8.0
    }
    
    # 事件数据
    calendar_data = pd.DataFrame({
        'id': ['stadium_event'],
        'name': ['篮球比赛'],
        'type': ['sports_event'],
        'start_time': [pd.Timestamp.now() - pd.Timedelta(minutes=60)],
        'end_time': [pd.Timestamp.now() + pd.Timedelta(hours=2)],
        'attendance': [15000],
        'longitude': [116.397],
        'latitude': [39.908]
    })
    
    print(f"\n当前交通状况:")
    for segment, data in current_traffic.items():
        speed_ratio = data['current_speed'] / data['free_flow_speed']
        if speed_ratio < 0.3:
            status = "🔴 严重拥堵"
        elif speed_ratio < 0.6:
            status = "🟡 中度拥堵"
        else:
            status = "🟢 畅通"
        
        print(f"  {segment}: {data['current_speed']}km/h ({status})")
    
    print(f"\n天气状况:")
    print(f"  降雨强度: {weather_data['rain_intensity']}mm/h")
    print(f"  能见度: {weather_data['visibility']}m")
    print(f"  风速: {weather_data['wind_speed']}m/s")
    
    print(f"\n特殊事件:")
    print(f"  活动: {calendar_data.iloc[0]['name']}")
    print(f"  观众: {calendar_data.iloc[0]['attendance']:,}人")
    
    # 执行各项分析
    print(f"\n" + "-" * 40)
    print("分析结果:")
    
    # 天气影响分析
    weather_analyzer = WeatherAnalyzer()
    weather_impact, weather_factors = weather_analyzer.analyze_weather_impact(
        weather_data, pd.DataFrame(current_traffic).T
    )
    print(f"  天气影响: {weather_impact:.2f} ({len(weather_factors)}个影响因子)")
    
    # 事故检测
    incident_detector = IncidentDetector()
    # 这里简化处理，假设没有检测到事故
    incidents = []
    
    # 事件检测
    event_analyzer = EventAnalyzer()
    events = event_analyzer.detect_events(
        pd.DataFrame(current_traffic).T, calendar_data
    )
    print(f"  事件影响: 检测到{len(events)}个相关事件")
    
    # 风险评估
    risk_engine = RiskAssessmentEngine()
    
    # 构建风险评估输入
    assessment_causes = []
    
    # 添加天气原因
    if weather_impact > 0.3:
        assessment_causes.append(CongestionCause(
            'weather_impact', CauseType.WEATHER, (116.397, 39.908),
            weather_impact, time.time(), list(current_traffic.keys()),
            weather_impact, 0.8, f"恶劣天气影响: {', '.join(weather_factors)}"
        ))
    
    # 添加事件原因
    for event in events:
        assessment_causes.append(event)
    
    if assessment_causes:
        risk_assessment = risk_engine.assess_risk(assessment_causes, [], {})
        print(f"  综合风险: {risk_assessment.risk_score:.2f} ({risk_assessment.overall_risk_level.name})")
        print(f"  预警等级: {risk_assessment.alert_level}")
        print(f"  主要建议:")
        for i, rec in enumerate(risk_assessment.recommendations[:3], 1):
            print(f"    {i}. {rec}")
    else:
        print(f"  综合风险: 0.10 (LOW)")
        print(f"  预警等级: green")
        print(f"  主要建议: 继续监控交通状况")


def main():
    """主函数"""
    print("拥堵原因分析器核心功能演示")
    print("=" * 80)
    print("本演示展示了拥堵原因分析器的主要功能:")
    print("1. 天气影响分析")
    print("2. 交通事故检测")
    print("3. 特殊事件影响分析")
    print("4. 因果关系推理")
    print("5. 传播路径分析")
    print("6. 时间序列分析")
    print("7. 综合风险评估")
    print("8. 综合场景分析")
    
    try:
        # 运行各个功能演示
        demo_weather_analysis()
        demo_incident_detection()
        demo_event_analysis()
        demo_causal_inference()
        demo_propagation_analysis()
        demo_time_series_analysis()
        demo_risk_assessment()
        demo_comprehensive_scenario()
        
        print("\n" + "=" * 80)
        print("🎉 所有功能演示完成!")
        print("=" * 80)
        
        print("\n拥堵原因分析器具有以下核心能力:")
        print("✓ 多维度原因分析（天气、事故、事件等）")
        print("✓ 因果关系推理和影响因子排序")
        print("✓ 拥堵传播路径分析")
        print("✓ 时间序列分析和趋势预测")
        print("✓ 风险评估和预警机制")
        print("✓ 科学的分析结果和建议生成")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()