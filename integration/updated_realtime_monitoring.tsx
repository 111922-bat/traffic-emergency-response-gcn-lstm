import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, CheckCircle, Clock, Wifi, WifiOff } from 'lucide-react';
import DataCard from './DataCard';
import TrafficHeatmap from './TrafficHeatmap';
import SegmentList from './SegmentList';
import { useApiService, useWebSocketService } from '../services/apiIntegration';

interface TrafficSegment {
  segment_id: string;
  speed: number;
  flow: number;
  density: number;
  occupancy: number;
  status: 'normal' | 'warning' | 'congested';
  lat: number;
  lng: number;
  name?: string;
}

interface RealtimeData {
  timestamp: string;
  total_segments: number;
  congested_segments: number;
  congestion_distance: number;
  average_speed: number;
  total_flow: number;
  system_status: string;
  segments: TrafficSegment[];
  weather: {
    condition: string;
    temperature: number;
    visibility: string;
    impact_factor: number;
  };
  traffic_light_status: {
    total_lights: number;
    green_lights: number;
    yellow_lights: number;
    red_lights: number;
  };
}

const RealTimeMonitoring: React.FC = () => {
  const { apiService, isConnected: apiConnected, error: apiError } = useApiService();
  const { 
    wsService, 
    isConnected: wsConnected, 
    realtimeData: wsRealtimeData, 
    subscribeTrafficData 
  } = useWebSocketService();

  const [realtimeData, setRealtimeData] = useState<RealtimeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('connecting');

  // 获取实时数据
  const fetchRealtimeData = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      
      // 优先使用WebSocket数据
      if (wsRealtimeData) {
        setRealtimeData(wsRealtimeData);
        setLastUpdate(new Date());
        setLoading(false);
        setConnectionStatus('connected');
        return;
      }

      // 使用API获取数据
      const response = await apiService.getRealtimeData();
      
      if (response.success && response.data) {
        setRealtimeData(response.data);
        setLastUpdate(new Date());
        setLoading(false);
        setConnectionStatus('connected');
      } else {
        throw new Error(response.error || '获取数据失败');
      }
    } catch (error) {
      console.error('获取实时数据失败:', error);
      setConnectionStatus('disconnected');
      setLoading(false);
      
      // 使用缓存数据作为降级
      const cachedData = localStorage.getItem('cached_realtime_data');
      if (cachedData) {
        try {
          const parsedData = JSON.parse(cachedData);
          setRealtimeData(parsedData);
          setConnectionStatus('disconnected');
        } catch (cacheError) {
          console.error('缓存数据解析失败:', cacheError);
        }
      }
    }
  }, [apiService, wsRealtimeData]);

  // 初始化数据获取
  useEffect(() => {
    fetchRealtimeData();
  }, [fetchRealtimeData]);

  // 定期刷新数据
  useEffect(() => {
    const interval = setInterval(fetchRealtimeData, 10000); // 每10秒刷新一次
    
    return () => clearInterval(interval);
  }, [fetchRealtimeData]);

  // 订阅WebSocket实时数据
  useEffect(() => {
    if (wsConnected) {
      subscribeTrafficData();
    }
  }, [wsConnected, subscribeTrafficData]);

  // 缓存数据到本地存储
  useEffect(() => {
    if (realtimeData && connectionStatus === 'connected') {
      localStorage.setItem('cached_realtime_data', JSON.stringify(realtimeData));
    }
  }, [realtimeData, connectionStatus]);

  // 连接状态指示器
  const getConnectionStatusIcon = () => {
    if (connectionStatus === 'connected') {
      return <Wifi className="w-4 h-4 text-green-500" />;
    } else if (connectionStatus === 'connecting') {
      return <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />;
    } else {
      return <WifiOff className="w-4 h-4 text-red-500" />;
    }
  };

  const getConnectionStatusText = () => {
    if (connectionStatus === 'connected') {
      return wsConnected ? '实时连接' : 'API连接';
    } else if (connectionStatus === 'connecting') {
      return '连接中...';
    } else {
      return '连接断开';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return 'text-semantic-success';
      case 'warning': return 'text-semantic-warning';
      case 'congested': return 'text-semantic-error';
      default: return 'text-text-secondary';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'normal': return CheckCircle;
      case 'warning': return AlertCircle;
      case 'congested': return AlertCircle;
      default: return CheckCircle;
    }
  };

  // 错误重试
  const handleRetry = useCallback(() => {
    setLoading(true);
    fetchRealtimeData();
  }, [fetchRealtimeData]);

  // 加载状态
  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-base mx-auto mb-4"></div>
          <p className="text-text-secondary">正在加载实时数据...</p>
        </div>
      </div>
    );
  }

  // 错误状态
  if (!realtimeData) {
    return (
      <div className="text-center text-text-secondary py-12">
        <AlertCircle className="w-12 h-12 mx-auto mb-4 text-semantic-error" />
        <p className="mb-4">无法获取实时数据</p>
        {apiError && <p className="text-sm text-semantic-error mb-4">错误: {apiError}</p>}
        <button
          onClick={handleRetry}
          className="px-4 py-2 bg-primary-base text-white rounded-lg hover:bg-primary-dark transition-colors"
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 连接状态栏 */}
      <div className="bg-bg-card rounded-lg p-4 border border-border-subtle">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getConnectionStatusIcon()}
            <span className="text-sm text-text-secondary">
              {getConnectionStatusText()}
            </span>
            {connectionStatus === 'disconnected' && (
              <span className="text-xs text-semantic-warning">
                (使用缓存数据)
              </span>
            )}
          </div>
          <div className="flex items-center space-x-4 text-sm text-text-secondary">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4" />
              <span>最后更新: {lastUpdate.toLocaleTimeString('zh-CN')}</span>
            </div>
            {connectionStatus === 'connected' && (
              <div className="w-2 h-2 bg-primary-base rounded-full animate-pulse"></div>
            )}
          </div>
        </div>
      </div>

      {/* 顶部概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <DataCard
          title="总拥堵里程"
          value={`${realtimeData.congestion_distance.toFixed(1)} km`}
          unit="拥堵路段"
          trend={realtimeData.congested_segments > 10 ? 'up' : 'down'}
          trendValue={`${realtimeData.congested_segments} 路段`}
          status={realtimeData.congested_segments > 15 ? 'warning' : 'normal'}
          icon={realtimeData.congested_segments > 15 ? AlertCircle : CheckCircle}
        />
        
        <DataCard
          title="平均速度"
          value={realtimeData.average_speed.toFixed(1)}
          unit="km/h"
          trend={realtimeData.average_speed > 50 ? 'up' : 'down'}
          trendValue={`${realtimeData.average_speed > 50 ? '良好' : '缓慢'}`}
          status={realtimeData.average_speed > 40 ? 'normal' : 'warning'}
          icon={TrendingUp}
        />
        
        <DataCard
          title="实时流量"
          value={Math.round(realtimeData.total_flow / 1000)}
          unit="k veh/h"
          trend="up"
          trendValue="持续增长"
          status="normal"
          icon={TrendingUp}
        />
        
        <DataCard
          title="系统状态"
          value={realtimeData.system_status}
          unit=""
          trend={realtimeData.system_status === '告警' ? 'down' : 'up'}
          trendValue={realtimeData.system_status === '告警' ? "需要关注" : "运行良好"}
          status={realtimeData.system_status === '告警' ? 'error' : 'normal'}
          icon={realtimeData.system_status === '告警' ? AlertCircle : CheckCircle}
        />
      </div>

      {/* 天气和信号灯状态 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-bg-card rounded-large p-6 border border-border-subtle">
          <h3 className="text-title text-text-primary mb-4">天气状况</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-text-secondary">天气</span>
              <span className="text-text-primary">{realtimeData.weather.condition}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">温度</span>
              <span className="text-text-primary">{realtimeData.weather.temperature}°C</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">能见度</span>
              <span className="text-text-primary">{realtimeData.weather.visibility}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">影响因子</span>
              <span className="text-text-primary">{realtimeData.weather.impact_factor}</span>
            </div>
          </div>
        </div>

        <div className="bg-bg-card rounded-large p-6 border border-border-subtle">
          <h3 className="text-title text-text-primary mb-4">信号灯状态</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-text-secondary">总数量</span>
              <span className="text-text-primary">{realtimeData.traffic_light_status.total_lights}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">绿灯</span>
              <span className="text-semantic-success">{realtimeData.traffic_light_status.green_lights}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">黄灯</span>
              <span className="text-semantic-warning">{realtimeData.traffic_light_status.yellow_lights}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-secondary">红灯</span>
              <span className="text-semantic-error">{realtimeData.traffic_light_status.red_lights}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 中央热力图 */}
      <div className="bg-bg-card rounded-large p-6 border border-border-subtle">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-title text-text-primary">实时路况热力图</h2>
          <div className="flex items-center space-x-4 text-small text-text-secondary">
            <div className="flex items-center space-x-2">
              <Clock className="w-4 h-4" />
              <span>最后更新: {lastUpdate.toLocaleTimeString('zh-CN')}</span>
            </div>
            {connectionStatus === 'connected' && (
              <div className="w-2 h-2 bg-primary-base rounded-full animate-pulse"></div>
            )}
          </div>
        </div>
        <TrafficHeatmap segments={realtimeData.segments} />
      </div>

      {/* 底部关键路段列表 */}
      <div className="bg-bg-card rounded-large p-6 border border-border-subtle">
        <h2 className="text-title text-text-primary mb-4">关键路段状态</h2>
        <SegmentList segments={realtimeData.segments.slice(0, 10)} />
      </div>

      {/* 连接问题提示 */}
      {connectionStatus === 'disconnected' && (
        <div className="bg-semantic-warning bg-opacity-10 border border-semantic-warning rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 text-semantic-warning" />
            <span className="text-semantic-warning font-medium">连接中断</span>
          </div>
          <p className="text-sm text-text-secondary mt-2">
            实时数据连接已断开，正在显示缓存数据。请检查网络连接或刷新页面重试。
          </p>
        </div>
      )}
    </div>
  );
};

export default RealTimeMonitoring;
