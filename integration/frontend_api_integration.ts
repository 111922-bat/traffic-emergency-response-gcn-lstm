/**
 * 前端API集成服务
 * Frontend API Integration Service for Intelligent Traffic Flow Prediction System
 * 
 * 实现前后端API连接、WebSocket实时通信、数据同步等功能
 */

import { io, Socket } from 'socket.io-client';

// API配置
export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:3001',
  WS_URL: process.env.REACT_APP_WS_URL || 'http://localhost:3001',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
};

// 基础API响应接口
export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp?: string;
}

// 交通数据接口
export interface TrafficSegment {
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

export interface RealtimeTrafficData {
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

// 预测数据接口
export interface PredictionRequest {
  incident_location: [number, number];
  prediction_time: number;
  impact_radius: number;
}

export interface PredictionStep {
  time: string;
  predicted_speed: number;
  predicted_flow: number;
  congestion_level: number;
  affected_segments: number;
  confidence: number;
}

export interface PredictionResult {
  prediction_id: string;
  incident_location: [number, number];
  prediction_time: number;
  impact_radius: number;
  prediction_steps: PredictionStep[];
  confidence_analysis: {
    short_term: {
      time_range: string;
      confidence: number;
      factors: string[];
    };
    medium_term: {
      time_range: string;
      confidence: number;
      factors: string[];
    };
    long_term: {
      time_range: string;
      confidence: number;
      factors: string[];
    };
  };
  model_info: {
    model_type: string;
    version: string;
    training_data: string;
    last_updated: string;
    accuracy_metrics: {
      mae: number;
      rmse: number;
      mape: number;
      r2: number;
    };
  };
}

// 应急车辆接口
export interface EmergencyVehicle {
  id: string;
  type: string;
  name: string;
  location: {
    lat: number;
    lng: number;
  };
  status: 'available' | 'busy' | 'on_scene';
  capacity: number;
  equipment: string[];
  response_time: number;
}

export interface DispatchRequest {
  incident_id: string;
  vehicle_id: string;
  location: [number, number];
}

export interface DispatchResult {
  dispatch_id: string;
  incident_id: string;
  vehicle_id: string;
  status: string;
  estimated_arrival: string;
  route_info: {
    distance: string;
    traffic_conditions: string;
    alternative_routes: number;
  };
  timestamp: string;
}

// 系统指标接口
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  gpu_usage: number;
  network_latency: number;
  model_performance: {
    mae: number;
    rmse: number;
    mape: number;
    r2: number;
    inference_time: number;
  };
  system_health: string;
  timestamp: string;
}

// 系统日志接口
export interface SystemLog {
  timestamp: string;
  level: 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  detail: string;
}

// API服务类
export class ApiService {
  private baseURL: string;
  private timeout: number;

  constructor(config = API_CONFIG) {
    this.baseURL = config.BASE_URL;
    this.timeout = config.TIMEOUT;
  }

  // 通用请求方法
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url, {
        ...defaultOptions,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API request failed: ${endpoint}`, error);
      throw error;
    }
  }

  // 健康检查
  async healthCheck(): Promise<ApiResponse> {
    return this.request('/api/health');
  }

  // 获取实时交通数据
  async getRealtimeData(): Promise<ApiResponse<RealtimeTrafficData>> {
    return this.request('/api/realtime');
  }

  // 交通流预测
  async predictTraffic(request: PredictionRequest): Promise<ApiResponse<PredictionResult>> {
    return this.request('/api/predict', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // 获取应急车辆信息
  async getEmergencyVehicles(): Promise<ApiResponse<EmergencyVehicle[]>> {
    return this.request('/api/emergency/vehicles');
  }

  // 调度应急车辆
  async dispatchVehicle(request: DispatchRequest): Promise<ApiResponse<DispatchResult>> {
    return this.request('/api/emergency/dispatch', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // 获取系统指标
  async getSystemMetrics(): Promise<ApiResponse<SystemMetrics>> {
    return this.request('/api/system/metrics');
  }

  // 获取系统日志
  async getSystemLogs(): Promise<ApiResponse<SystemLog[]>> {
    return this.request('/api/system/logs');
  }
}

// WebSocket服务类
export class WebSocketService {
  private socket: Socket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Function[]> = new Map();

  constructor(url = API_CONFIG.WS_URL) {
    this.url = url;
  }

  // 连接WebSocket
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.socket = io(this.url, {
          transports: ['websocket', 'polling'],
          timeout: 5000,
        });

        this.socket.on('connect', () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          this.emit('connected');
          resolve();
        });

        this.socket.on('disconnect', (reason) => {
          console.log('WebSocket disconnected:', reason);
          this.emit('disconnected', reason);
          this.handleReconnect();
        });

        this.socket.on('connect_error', (error) => {
          console.error('WebSocket connection error:', error);
          this.emit('connection_error', error);
          reject(error);
        });

        // 订阅实时交通数据
        this.socket.on('traffic-data', (data: RealtimeTrafficData) => {
          this.emit('traffic_data', data);
        });

        this.socket.on('subscription_confirmed', (data) => {
          console.log('Subscription confirmed:', data);
        });

      } catch (error) {
        reject(error);
      }
    });
  }

  // 断开WebSocket连接
  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // 订阅实时交通数据
  subscribeTrafficData(): void {
    if (this.socket) {
      this.socket.emit('subscribe_traffic_data');
    }
  }

  // 事件监听
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  // 移除事件监听
  off(event: string, callback?: Function): void {
    if (!callback) {
      this.listeners.delete(event);
      return;
    }

    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      const index = eventListeners.indexOf(callback);
      if (index > -1) {
        eventListeners.splice(index, 1);
      }
    }
  }

  // 触发事件
  private emit(event: string, ...args: any[]): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => callback(...args));
    }
  }

  // 处理重连
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }, this.reconnectDelay * this.reconnectAttempts);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed');
    }
  }

  // 检查连接状态
  isConnected(): boolean {
    return this.socket?.connected || false;
  }
}

// 数据缓存服务
export class DataCache {
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private defaultTTL = 5 * 60 * 1000; // 5分钟

  // 设置缓存
  set(key: string, data: any, ttl = this.defaultTTL): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  // 获取缓存
  get(key: string): any | null {
    const cached = this.cache.get(key);
    if (!cached) {
      return null;
    }

    const now = Date.now();
    if (now - cached.timestamp > cached.ttl) {
      this.cache.delete(key);
      return null;
    }

    return cached.data;
  }

  // 清除缓存
  clear(key?: string): void {
    if (key) {
      this.cache.delete(key);
    } else {
      this.cache.clear();
    }
  }

  // 清理过期缓存
  cleanup(): void {
    const now = Date.now();
    for (const [key, cached] of this.cache.entries()) {
      if (now - cached.timestamp > cached.ttl) {
        this.cache.delete(key);
      }
    }
  }
}

// 集成服务管理器
export class IntegrationService {
  private apiService: ApiService;
  private wsService: WebSocketService;
  private dataCache: DataCache;
  private isInitialized = false;

  constructor() {
    this.apiService = new ApiService();
    this.wsService = new WebSocketService();
    this.dataCache = new DataCache();
  }

  // 初始化服务
  async initialize(): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    try {
      console.log('初始化集成服务...');

      // 检查API健康状态
      const health = await this.apiService.healthCheck();
      if (!health.success) {
        throw new Error('API服务不可用');
      }

      // 连接WebSocket
      await this.wsService.connect();
      this.wsService.subscribeTrafficData();

      // 启动缓存清理
      setInterval(() => {
        this.dataCache.cleanup();
      }, 60000); // 每分钟清理一次

      this.isInitialized = true;
      console.log('集成服务初始化完成');
    } catch (error) {
      console.error('集成服务初始化失败:', error);
      throw error;
    }
  }

  // 获取API服务实例
  getApiService(): ApiService {
    return this.apiService;
  }

  // 获取WebSocket服务实例
  getWebSocketService(): WebSocketService {
    return this.wsService;
  }

  // 获取缓存服务实例
  getCacheService(): DataCache {
    return this.dataCache;
  }

  // 关闭服务
  async shutdown(): Promise<void> {
    this.wsService.disconnect();
    this.dataCache.clear();
    this.isInitialized = false;
    console.log('集成服务已关闭');
  }
}

// React Hook for API integration
import { useState, useEffect, useCallback } from 'react';

export function useApiService() {
  const [apiService] = useState(() => new ApiService());
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const health = await apiService.healthCheck();
        setIsConnected(health.success);
        setError(null);
      } catch (err) {
        setIsConnected(false);
        setError(err instanceof Error ? err.message : '连接失败');
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 30000); // 每30秒检查一次

    return () => clearInterval(interval);
  }, [apiService]);

  return { apiService, isConnected, error };
}

export function useWebSocketService() {
  const [wsService] = useState(() => new WebSocketService());
  const [isConnected, setIsConnected] = useState(false);
  const [realtimeData, setRealtimeData] = useState<RealtimeTrafficData | null>(null);

  useEffect(() => {
    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);
    const handleTrafficData = (data: RealtimeTrafficData) => setRealtimeData(data);

    wsService.on('connected', handleConnect);
    wsService.on('disconnected', handleDisconnect);
    wsService.on('traffic_data', handleTrafficData);

    wsService.connect().catch(console.error);

    return () => {
      wsService.off('connected', handleConnect);
      wsService.off('disconnected', handleDisconnect);
      wsService.off('traffic_data', handleTrafficData);
      wsService.disconnect();
    };
  }, [wsService]);

  const subscribeTrafficData = useCallback(() => {
    wsService.subscribeTrafficData();
  }, [wsService]);

  return { wsService, isConnected, realtimeData, subscribeTrafficData };
}

// 导出默认实例
export const integrationService = new IntegrationService();
export const apiService = new ApiService();
export const wsService = new WebSocketService();
export const dataCache = new DataCache();
