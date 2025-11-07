# 交通流预测系统API接口规范(面向工程落地与可解释预测)

## 1. 引言与总体目标

城市交通系统正在快速走向数据驱动与智能决策,管理者与出行者对拥堵预警、路径优化与应急处置的实时性与可解释性提出了更高要求。为此,本规范提出一套面向工程落地与可解释预测的交通流预测系统API接口标准,目标是在统一协议下,提供从实时数据接入、拥堵预测、路径规划到自然语言解释与系统管理的全链路能力。

本规范覆盖以下接口族:
- 预测服务API:输入事故位置与交通数据,输出拥堵时空预测与解释。
- 路径规划API:输入起终点与偏好,输出最优路径与备选方案。
- 实时数据API:提供交通状态订阅与查询能力。
- LLM解释API:生成面向人读的拥堵成因、传播路径与建议。
- 系统管理API:模型版本管理、数据接入与质量监控、运维与审计。

本规范适用于后端API设计师、交通系统架构师、数据科学家与运维/安全工程师,强调一致的数据口径、明确的性能SLO、稳定的版本策略与完善的安全认证机制。我们以公开数据集与工程实践为依据,确保接口在真实生产环境中的可复现性与可运维性[^1][^2][^3][^4]。

信息缺口与处理:鉴于跨城统一评测协议与具体SLA数值尚需落地确认,本规范先给出相对指标与工程口径,具体数值由部署方在生产环境中标定与确认。

---

## 2. 术语、数据口径与单位

为避免歧义,本规范对关键术语、数据口径与单位进行统一。

- 路网元素
  - 节点(Node):传感器/路段标识或拓扑交叉点。
  - 边(Edge):道路路段,具有方向、长度与车道数。
  - 路段(Segment):道路的连续区间,通常与传感器或拓扑边对应。
- 交通状态
  - 速度(V):单位km/h或m/s,建议统一为km/h。
  - 流量(Q):单位veh/h或veh/5min,建议veh/h。
  - 密度/占有率(K):单位veh/km或%,建议veh/km。
- 时间与窗口
  - 时间戳:ISO 8601,UTC或本地时区(接口统一UTC,响应可带时区)。
  - 采样间隔:5分钟为基准(与METR-LA/PEMS一致)。
  - 历史窗口:默认12步(60分钟),预测步长默认3步(15分钟)。
- 空间口径
  - 坐标系:WGS84或本地投影坐标,接口统一WGS84,内部转换由服务端处理。
  - 路段编码:统一路段ID体系,支持跨城迁移。

为便于工程落地,表1给出统一字段与单位口径。

表1 统一字段与单位口径表(示例)

| 字段 | 含义 | 类型 | 单位 | 示例 | 备注 |
|---|---|---|---|---|---|
| timestamp | 数据时间戳 | string (ISO 8601) | UTC | 2025-11-05T19:52:55Z | 统一UTC,响应可带偏移 |
| segment_id | 路段唯一ID | string | - | CN_SH_000123 | 全局唯一 |
| speed | 断面速度 | number | km/h | 52.3 | ≥0 |
| flow | 断面流量 | number | veh/h | 1200 | ≥0 |
| occupancy | 占有率 | number | % | 35.0 | 0–100 |
| geom | 路段几何 | GeoJSON LineString | - | {"type":"LineString","coordinates":[[...]]} | WGS84 |
| weather | 天气状态 | object | - | {"temp":15.2,"precip":0.0} | 可选 |
| incident | 事件信息 | object | - | {"type":"accident","lanes_blocked":2} | 可选 |

参考数据集与口径:METR-LA/PEMS-BAY/PeMSD7等采用5分钟采样,速度单位为km/h,任务定义以“过去12步预测未来12/3步”为常见设置[^1][^2]。

---

## 3. 系统总体架构与API族设计

本系统采用分层架构,确保接口职责清晰与演进有序。

- 数据接入层:标准化接入检测器、轨迹、事件与信号日志,统一时空对齐与口径。
- 时空建模层:融合物理约束(LWR/CTM)与数据驱动(GCN+LSTM/Transformer),支持动态图建模与在线更新[^3][^4]。
- 预测与路径层:提供拥堵预测、传播范围估计与路径规划(滚动时域)。
- 解释与呈现层:通过LLM解释API生成自然语言说明与建议。
- 管理与安全层:模型管理、数据管理、运维审计与安全认证授权。

接口族与职责映射如表2。

表2 API族与职责映射表

| 接口族 | 主要端点 | 输入 | 输出 | 核心SLO | 依赖 |
|---|---|---|---|---|---|
| 预测服务 | /predict, /batch-predict | 事故、交通数据、窗口参数 | 拥堵时空预测、解释 | 相对延迟等级P1/P2/P3 | 实时数据、模型注册 |
| 路径规划 | /route, /recalculate | 起终点、偏好、约束 | 最优路径、备选方案 | 相对延迟等级P1/P2/P3 | 预测服务、路网拓扑 |
| 实时数据 | /realtime/subscribe, /query | 订阅条件或segment_id | 流式状态、快照 | 相对延迟等级P1 | 数据接入层 |
| LLM解释 | /explain | 预测结果、因果要素 | 自然语言解释 | 相对延迟等级P1/P2 | 预测服务、因果要素 |
| 系统管理 | /admin/models, /admin/data | 模型/数据管理请求 | 管理确认与状态 | 相对延迟等级P2 | 注册表、审计 |

版本策略:URL前缀采用/v1;向后兼容与弃用流程遵循“新增非破坏字段→小版本升级→废弃公告→停用”的节奏,保留足够迁移窗口。

---

## 4. 通用API约定与错误模型

本系统遵循REST风格,统一JSON与GeoJSON响应结构,支持分页与流式事件(SSE)或WebSocket。

- 资源定位:/v1/{resource}/{action}
- 认证:OAuth2.0 Client Credentials或API Key(HMAC签名可选)
- 幂等:对写操作要求Idempotency-Key
- 重试:客户端遵循指数退避与幂等保障
- 错误模型:统一错误码与HTTP状态映射,响应体包含trace_id便于审计

表3 错误码与HTTP状态映射(示例)

| 错误码 | HTTP状态 | 说明 | 客户端建议 |
|---|---|---|---|
| 0_OK | 200 | 成功 | - |
| 1001_INVALID_ARGUMENT | 400 | 参数不合法 | 修正参数后重试 |
| 1002_MISSING_ARGUMENT | 400 | 缺少必要参数 | 补齐参数 |
| 1003_OUT_OF_RANGE | 422 | 参数超范围 | 调整窗口/阈值 |
| 1004_ALREADY_EXISTS | 409 | 资源已存在 | 使用幂等键或更换ID |
| 2001_UNAUTHENTICATED | 401 | 认证失败 | 刷新令牌/密钥 |
| 2003_FORBIDDEN | 403 | 无权限 | 申请权限或更换账号 |
| 3001_NOT_FOUND | 404 | 资源不存在 | 校验ID/版本 |
| 4001_INTERNAL_ERROR | 500 | 服务器内部错误 | 重试并联系支持 |
| 4002_SERVICE_UNAVAILABLE | 503 | 服务暂不可用 | 稍后重试/降级 |
| 4003_DEADLINE_EXCEEDED | 504 | 超时 | 缩小范围/优化参数 |

审计字段:所有响应包含trace_id;管理操作记录actor、action、resource、before/after快照与时间戳。

---

## 5. 预测服务API设计(输入事故位置、交通数据,输出拥堵预测)

本接口将事故与交通状态输入转换为未来15–60分钟的拥堵时空预测,提供可解释要素(传播速度、影响范围、持续时长、概率转移)与自然语言解释。

输入规范:
- 事故位置:GeoJSON Point或路段ID(segment_id)。
- 交通数据:历史窗口(默认12步,5分钟间隔),包含速度/流量/占有率。
- 外生变量:天气(可选)、节假日(可选)、信号相位(可选)。
- 窗口参数:input_len、prediction_steps(默认3步)、步长(5分钟)。

输出规范:
- 拥堵时空预测:路段级速度/密度/占有率的时间序列。
- 传播要素:传播速度c、影响范围S、持续时长T、传播概率P(可选)。
- 解释字段:拥堵成因摘要、传播路径与建议。
- 版本与模型元信息:model_id、version、training_data_range、metrics。

算法选择与工程口径:
- 稳态与短序列:GCN+LSTM(工程稳健、延迟可控)。
- 长序列与复杂依赖:GCN+TCN/Transformer(并行与长程优势)。
- 动态图:DGCN/AST-DGCN(时变邻接与时空联合建模)。
- 物理约束:CTM/LWR先验与矩形法用于可解释预测与快速评估[^3][^4][^5][^6][^7][^8][^9]。

表4 预测输入字段表(示例)

| 字段 | 类型 | 必填 | 单位 | 示例 | 校验 |
|---|---|---|---|---|---|
| incident.location | GeoJSON Point 或 segment_id | 是 | - | {"type":"Point","coordinates":[121.5,31.2]} 或 "CN_SH_000123" | 坐标系WGS84 |
| traffic_history | array of object | 是 | - | [{"timestamp":"...","segment_id":"...","speed":...,"flow":...,"occupancy":...}] | 12步一致口径 |
| weather | object | 否 | - | {"temp":15.2,"precip":0.0} | 字段可选 |
| holiday | boolean | 否 | - | false | - |
| signal_phase | object | 否 | - | {"segment_id":"...","phase":"green","remaining":30} | 可选 |
| window.input_len | integer | 否 | - | 12 | 5–30 |
| window.prediction_steps | integer | 否 | - | 3 | 1–12 |

表5 预测输出字段表(示例)

| 字段 | 类型 | 单位 | 说明 |
|---|---|---|---|
| predictions[].segment_id | string | - | 路段ID |
| predictions[].speed_forecast | array<number> | km/h | 未来速度序列 |
| predictions[].density_forecast | array<number> | veh/km | 未来密度序列 |
| explain.c | number | km/h | 传播速度估计 |
| explain.S | string | - | 影响范围描述(网格/节点集合) |
| explain.T | number | min | 持续时长估计 |
| explain.P | number | 0–1 | 传播概率(可选) |
| explain.summary | string | - | 成因摘要与建议 |
| model.meta | object | - | model_id、version、metrics |

表6 预测任务参数与默认值

| 参数 | 默认值 | 范围 | 说明 |
|---|---|---|---|
| input_len | 12 | 5–30 | 历史窗口步数(5分钟/步) |
| prediction_steps | 3 | 1–12 | 预测步数(5分钟/步) |
| step_interval | 5 | 5 | 分钟 |
| model_family | GCN+LSTM | - | 可选GCN+TCN/Transformer |
| dynamic_graph | false | true/false | 是否启用动态图 |
| use_physics_prior | false | true/false | 是否融合CTM/LWR |

性能与SLO:
- 延迟等级:P1(低延迟,默认)、P2(平衡)、P3(高质量)。不同等级映射不同模型与缓存策略。
- 吞吐:支持批量预测(/batch-predict),按提交窗口与资源配额调度。
- 缓存:节点嵌入与注意力权重可缓存并增量刷新;动态图场景优先缓存。

表7 预测质量指标与口径(示例)

| 指标 | 定义 | 口径 |
|---|---|---|
| MAE | 平均绝对误差 | 速度(km/h) |
| RMSE | 均方根误差 | 速度(km/h) |
| MAPE | 平均绝对百分比误差 | 速度,零值平滑 |
| R² | 决定系数 | 拟合优度 |

注:具体数值依赖数据集与切分,工程口径需与离线评测协议一致[^1][^2]。

### 5.1 输入与输出Schema(JSON/GeoJSON)

- 输入:incident.location支持GeoJSON Point或segment_id;traffic_history为数组,每项包含timestamp、segment_id、speed、flow、occupancy。
- 输出:predictions为数组,每项包含segment_id与速度/密度时间序列;explain包含c、S、T、P与summary。
- GeoJSON:路段geom以LineString表示;拥堵热力图可返回GeoJSON FeatureCollection(可选)。

### 5.2 预测质量与误差口径

- 指标定义与注意事项如表8。
- 零值处理:MAPE在速度接近零时不稳定,需阈值平滑或采用MAE替代。
- 评测协议:统一train/val/test切分与归一化,避免数据泄漏。

表8 评测指标定义与注意事项

| 指标 | 定义 | 注意事项 |
|---|---|---|
| MAE | 平均绝对误差 | 业务可解释性强 |
| RMSE | 均方根误差 | 对大误差敏感 |
| MAPE | 平均绝对百分比误差 | 零值需平滑处理 |
| R² | 决定系数 | 越接近1越好 |

### 5.3 性能与扩展性

- 批处理:支持批量预测,提交窗口与配额控制。
- 缓存:节点嵌入与注意力权重缓存;动态图增量更新。
- 并行化:多线程/多进程与异步队列。

表9 性能SLO与资源配额(示例)

| 等级 | 目标 | 资源配额 | 适用场景 |
|---|---|---|---|
| P1 | 低延迟 | 中等CPU/内存 | 在线预警 |
| P2 | 平衡 | 中高CPU/内存 | 常规预测 |
| P3 | 高质量 | 高CPU/内存 | 离线评估 |

---

## 6. 路径规划API设计(输入起终点,输出最优路径)

本接口在路网拓扑上计算最优路径,支持多目标(时间、拥堵风险、收费、应急车道避让)与滚动时域重规划。

输入:
- 起终点:经纬度或路段ID。
- 偏好:时间优先、拥堵风险优先、收费避让、应急车道避让。
- 约束:车辆类型、应急通行、车道限制。
- 动态权重:由预测服务提供的拥堵风险与传播概率加权。

输出:
- 主路径:路段序列与预计到达时间(ETA)。
- 备选方案:若干备选路径与比较指标(时间、风险、可靠性)。
- 路径解释:关键决策点、拥堵风险点与绕行理由。

算法策略:
- 动态最短路:结合实时与预测速度场。
- 滚动时域重规划:随预测更新调整路径。
- 因果诱导:避免将流量引入潜在拥堵区[^10][^11][^12]。

表10 路径规划输入参数表

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| origin | GeoJSON Point 或 segment_id | 是 | - | 起点 |
| destination | GeoJSON Point 或 segment_id | 是 | - | 终点 |
| preferences.time_weight | number | 否 | 0.5 | 时间权重(0–1) |
| preferences.risk_weight | number | 否 | 0.5 | 拥堵风险权重(0–1) |
| constraints.vehicle_type | string | 否 | "normal" | normal/emergency |
| constraints.lane_restrictions | array | 否 | [] | 车道限制列表 |
| dynamic.routing_horizon | integer | 否 | 15 | 分钟 |
| dynamic.replan_interval | integer | 否 | 5 | 分钟 |

表11 路径规划输出字段表

| 字段 | 类型 | 说明 |
|---|---|---|
| primary_route.segment_ids | array<string> | 主路径路段序列 |
| primary_route.eta | number | 预计到达时间(分钟) |
| primary_route.geometry | GeoJSON LineString | 主路径几何 |
| alternatives[] | array<object> | 备选路径 |
| alternatives[].segment_ids | array<string> | 备选路段序列 |
| alternatives[].eta | number | 备选ETA |
| alternatives[].risk_score | number | 风险评分 |
| explanation | string | 关键决策点与绕行理由 |

表12 目标函数与权重配置示例

| 目标 | 权重 | 说明 |
|---|---|---|
| 时间 | 0.5 | 行程时间最短 |
| 风险 | 0.5 | 拥堵风险最低 |
| 收费 | 0.0 | 收费避让(可选) |
| 应急避让 | 0.0 | 应急车道避让(可选) |

### 6.1 约束与偏好建模

- 车辆类型:应急车辆优先,支持模糊优先级控制与信号优先协同。
- 收费与禁行:收费避让与车道限制作为约束或惩罚项。
- 应急车道:避让策略与路径解释中明确提示。

表13 约束与偏好枚举

| 名称 | 取值 | 说明 |
|---|---|---|
| vehicle_type | normal/emergency | 车辆类型 |
| toll_avoid | true/false | 收费避让 |
| emergency_lane_avoid | true/false | 应急车道避让 |
| lane_restrictions | array | 车道限制清单 |

### 6.2 动态重规划机制

- 触发条件:预测更新、事件变更、路况突变。
- 滚动时域:按replan_interval与routing_horizon执行。
- 稳定性与用户遵从:避免频繁变更,提供明确解释与置信度。

---

## 7. 实时数据API设计(获取实时交通状态)

本接口提供实时交通状态的订阅与查询,支持SSE或WebSocket,返回速度/流量/占有率等字段。

能力:
- 订阅:按路段ID、地理范围或事件类型订阅。
- 查询:按segment_id或空间范围返回快照。
- 频率:与采样间隔一致(默认5分钟),支持增量推送。

表14 实时数据字段与单位

| 字段 | 单位 | 说明 |
|---|---|---|
| speed | km/h | 断面速度 |
| flow | veh/h | 断面流量 |
| occupancy | % | 占有率 |
| timestamp | ISO 8601 | 数据时间戳 |
| geom | GeoJSON | 路段几何 |

表15 订阅/查询参数表

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| segments | array<string> | 否 | 路段ID列表 |
| bbox | GeoJSON Polygon | 否 | 空间范围 |
| event_types | array<string> | 否 | 事件类型过滤 |
| interval | integer | 否 | 推送间隔(秒) |

表16 数据质量与可用性字段

| 字段 | 说明 |
|---|---|
| missing_rate | 缺失率 |
| anomaly_score | 异常评分 |
| last_updated | 最近更新时间 |
| source | 数据源标识 |

---

## 8. LLM解释API设计(生成自然语言解释)

本接口将预测与规划结果转化为自然语言解释,帮助管理者与公众理解决策依据。

输入:
- 预测结果:时空预测与传播要素。
- 因果要素:传播速度c、影响范围S、持续时长T、概率转移P。
- 上下文:事故、天气、信号相位与历史模式。

输出:
- 解释文本:拥堵成因、传播路径、影响范围与建议。
- 语气与长度:专业版/公众版,短/中/长三档。
- 语言:中文(可扩展多语言)。

表17 LLM解释输入字段表

| 字段 | 类型 | 说明 |
|---|---|---|
| predictions | object | 时空预测结果 |
| explain.c | number | 传播速度 |
| explain.S | string | 影响范围 |
| explain.T | number | 持续时长 |
| explain.P | number | 传播概率 |
| context.incident | object | 事件信息 |
| context.weather | object | 天气 |
| context.signal_phase | object | 信号相位 |
| style.tone | string | 专业/公众 |
| style.length | string | 短/中/长 |
| language | string | zh/en(默认zh) |

表18 解释输出字段与格式

| 字段 | 类型 | 说明 |
|---|---|---|
| summary | string | 拥堵成因摘要 |
| propagation | string | 传播路径说明 |
| impact | string | 影响范围描述 |
| recommendations | string | 建议措施 |
| confidence | number | 解释置信度(0–1) |

解释生成策略与物理/规则先验融合:
- 将冲击波与CTM/LWR的物理先验、传播规则与因果要素作为提示输入,提升解释的可信度与可审计性[^13][^14][^15][^16]。

---

## 9. 系统管理API设计(模型管理、数据管理)

本接口提供模型版本管理、数据接入与质量监控、运维与审计能力。

模型管理:
- 注册:model_id、version、输入/输出Schema、训练数据范围、评测指标与延迟口径。
- 上线/下线:灰度与回滚策略。
- 资源配额:按版本分配CPU/内存与并发。

数据管理:
- 接入:检测器、轨迹、事件、信号日志接入与权限。
- 质量监控:缺失率、异常检测、漂移报警。
- 保留策略:分层存储与生命周期。

运维与审计:
- 审计日志:actor、action、resource、before/after、timestamp。
- 配额管理:租户/应用维度。
- 健康检查与告警:阈值与路由。

表19 模型元数据字段表

| 字段 | 说明 |
|---|---|
| model_id | 模型标识 |
| version | 版本号 |
| input_schema | 输入Schema摘要 |
| output_schema | 输出Schema摘要 |
| training_data_range | 训练数据时间范围 |
| metrics | 评测指标(MAE/RMSE/MAPE/R²) |
| latency_profile | 延迟等级(P1/P2/P3) |
| resource_quota | 资源配额 |

表20 数据接入与质量指标表

| 指标 | 说明 |
|---|---|
| missing_rate | 缺失率 |
| anomaly_score | 异常评分 |
| drift_flag | 漂移标记 |
| source | 数据源 |

表21 运维与审计事件字段

| 字段 | 说明 |
|---|---|
| actor | 操作人/服务 |
| action | 操作类型 |
| resource | 资源标识 |
| before | 变更前快照 |
| after | 变更后快照 |
| timestamp | 时间戳 |

模型注册与版本管理参考工程实践与开源实现(STGCN/T-GCN),确保输入/输出与延迟口径可追踪[^3][^4]。

---

## 10. API文档与开发者体验

文档结构:
- 概述:接口族与能力。
- 参考:端点、参数与示例。
- 示例:典型场景与错误处理。
- 变更日志:版本与弃用流程。

示例与SDK:
- 提供cURL、Python/JavaScript SDK示例。
- Postman/Insomnia集合与环境变量。

可观测性:
- trace_id贯穿请求。
- 指标与日志结构化输出。
- 错误注入与混沌测试指引。

表22 文档目录与示例索引

| 章节 | 内容 | 示例 |
|---|---|---|
| 概述 | 接口族与能力 | - |
| 预测服务 | 端点与Schema | 事故输入与预测输出 |
| 路径规划 | 端点与权重 | 起终点与重规划 |
| 实时数据 | 订阅与查询 | SSE/WebSocket示例 |
| LLM解释 | 提示与输出 | 专业/公众版 |
| 系统管理 | 模型与数据 | 注册与审计 |
| 变更日志 | 版本与弃用 | 迁移指南 |

---

## 11. 安全认证与授权机制

本系统采用OAuth2.0(客户端模式)为主,API Key(HMAC签名)为辅的双轨认证,支持细粒度RBAC与租户隔离。

- OAuth2.0:获取访问令牌(scope限定),适合服务到服务场景。
- API Key+HMAC:适用于轻量客户端,请求签名防篡改。
- RBAC:资源-动作-角色映射,支持审批流与审计。
- 租户隔离:数据与配额按租户维度隔离。
- 传输安全:TLS强制,敏感字段可选字段级加密。

表23 角色-权限矩阵(示例)

| 角色 | 资源 | 动作 | 权限 |
|---|---|---|---|
| admin | 模型/数据/审计 | 读/写/管理 | 全部 |
| analyst | 预测/解释 | 读/执行 | 预测与解释 |
| app | 实时数据/路径规划 | 读/执行 | 受限资源 |
| auditor | 审计日志 | 读 | 只读 |

表24 认证流程与令牌字段

| 字段 | 说明 |
|---|---|
| access_token | 访问令牌 |
| scope | 权限范围 |
| expires_in | 过期时间 |
| token_type | Bearer |

表25 安全事件与审计类型

| 事件 | 说明 |
|---|---|
| AUTH_SUCCESS | 认证成功 |
| AUTH_FAILURE | 认证失败 |
| ACCESS_DENIED | 访问拒绝 |
| RATE_LIMIT | 速率限制触发 |
| KEY_ROTATION | 密钥轮换 |

---

## 12. 性能、容量规划与SLA

目标设定采用相对等级与工程口径,具体数值由部署方标定。

- 延迟等级:P1(低延迟)、P2(平衡)、P3(高质量)。
- 吞吐:按批量与并发控制,支持流式。
- 容量:连接数、订阅数与存储增长预估。
- 降级:模型切换与缓存降级策略。

表26 性能目标与SLA(示例)

| 等级 | 延迟目标 | 吞吐 | 可用性 |
|---|---|---|---|
| P1 | 低延迟 | 中等 | 高 |
| P2 | 平衡 | 中高 | 高 |
| P3 | 高质量 | 中 | 高 |

表27 容量规划参数

| 参数 | 说明 |
|---|---|
| max_connections | 最大连接数 |
| max_subscriptions | 最大订阅数 |
| retention_days | 数据保留天数 |
| storage_growth | 存储增长预估 |

表28 降级与回退策略

| 场景 | 策略 |
|---|---|
| 模型不可用 | 切换至稳态基线(LSTM) |
| 缓存失效 | 降级至静态图与历史均值 |
| 负载过高 | 降低预测步长与并发 |

工程参考:GCN+LSTM的工程落地与训练配置为延迟与吞吐设定提供基线[^5]。

---

## 13. 测试、验收与持续改进

测试体系:
- 单元/集成/契约测试:确保接口与依赖稳定。
- 回归测试:覆盖版本升级与Schema变更。
- 混沌工程:验证故障与降级策略。

验收标准:
- 预测质量:MAE/RMSE/MAPE/R²达标。
- 性能:延迟与吞吐达标。
- 稳定性:错误率与重试成功率达标。

A/B测试:
- 指标:行程时间、延误、队列长度、应急响应时间。
- 流程:离线回放→在线试验→滚动优化。

持续改进:
- 指标看板:质量与性能监控。
- 告警:异常与漂移报警。
- 反馈闭环:用户与运维反馈驱动迭代。

表29 测试用例矩阵(示例)

| 场景 | 输入 | 期望输出 | 边界条件 |
|---|---|---|---|
| 事故输入 | 事故位置+历史窗口 | 预测与解释 | 缺失字段/异常值 |
| 路径规划 | 起终点+偏好 | 主路径与备选 | 禁行/应急车道 |
| 实时订阅 | 路段列表 | 流式状态 | 断线重连 |
| LLM解释 | 预测+因果要素 | 自然语言 | 风格与长度 |

表30 验收指标与阈值(示例)

| 指标 | 阈值 | 说明 |
|---|---|---|
| MAE | 业务设定 | 速度误差 |
| RMSE | 业务设定 | 大误差敏感 |
| MAPE | 业务设定 | 百分比误差 |
| R² | 业务设定 | 拟合优度 |
| 延迟等级 | P1/P2/P3 | 工程口径 |

表31 A/B测试指标与流程

| 指标 | 说明 |
|---|---|
| 行程时间 | 路径时间对比 |
| 延误 | 拥堵延误变化 |
| 队列长度 | 瓶颈排队长度 |
| 应急响应 | 应急车辆通行时间 |
| 流程 | 离线→在线→滚动优化 |

---

## 14. 附录:数据Schema与示例Payload

为便于实现,本附录给出核心Schema与示例。

表32 字段字典总表(节选)

| 名称 | 类型 | 单位 | 必填 | 示例 |
|---|---|---|---|---|
| timestamp | string | ISO 8601 | 是 | 2025-11-05T19:52:55Z |
| segment_id | string | - | 是 | CN_SH_000123 |
| speed | number | km/h | 是 | 52.3 |
| flow | number | veh/h | 是 | 1200 |
| occupancy | number | % | 是 | 35.0 |
| geom | GeoJSON | - | 是 | LineString |
| incident.location | GeoJSON/segment_id | - | 是 | Point或ID |
| window.input_len | integer | - | 否 | 12 |
| window.prediction_steps | integer | - | 否 | 3 |

示例:预测请求与响应

请求体(预测服务):
```json
{
  "incident": {
    "location": {"type":"Point","coordinates":[121.5,31.2]}
  },
  "traffic_history": [
    {"timestamp":"2025-11-05T19:00:00Z","segment_id":"CN_SH_000123","speed":55.0,"flow":1100,"occupancy":30.0},
    {"timestamp":"2025-11-05T19:05:00Z","segment_id":"CN_SH_000123","speed":53.0,"flow":1150,"occupancy":32.0}
    // ... 共12步
  ],
  "weather": {"temp":15.2,"precip":0.0},
  "window": {"input_len":12,"prediction_steps":3}
}
```

响应体(预测服务):
```json
{
  "predictions": [
    {"segment_id":"CN_SH_000123","speed_forecast":[50.2,48.1,46.5],"density_forecast":[12.5,13.2,14.1]}
  ],
  "explain": {
    "c": 8.5,
    "S": "节点{12-18}网格区域",
    "T": 35,
    "P": 0.72,
    "summary": "事故导致上游密度上升,形成后向波,预计35分钟恢复。"
  },
  "model": {
    "meta": {
      "model_id":"gcntcn_v1",
      "version":"2025.11",
      "training_data_range":"2025-01-01/2025-10-31",
      "metrics":{"MAE":2.1,"RMSE":3.4,"MAPE":0.055,"R2":0.91},
      "latency_profile":"P1"
    }
  }
}
```

示例:路径规划请求与响应

请求体(路径规划):
```json
{
  "origin": {"type":"Point","coordinates":[121.48,31.23]},
  "destination": {"type":"Point","coordinates":[121.52,31.25]},
  "preferences": {"time_weight":0.6,"risk_weight":0.4},
  "constraints": {"vehicle_type":"normal"},
  "dynamic": {"routing_horizon":15,"replan_interval":5}
}
```

响应体(路径规划):
```json
{
  "primary_route": {
    "segment_ids":["CN_SH_000120","CN_SH_000121","CN_SH_000122","CN_SH_000123"],
    "eta":18,
    "geometry":{"type":"LineString","coordinates":[[121.48,31.23],[121.49,31.24],[121.50,31.245],[121.52,31.25]]}
  },
  "alternatives":[
    {"segment_ids":["CN_SH_000130","CN_SH_000131","CN_SH_000123"],"eta":20,"risk_score":0.31}
  ],
  "explanation":"主线沿主干道,避让事故影响区,备选绕行距离略长但风险更低。"
}
```

示例:实时数据订阅与查询

SSE事件:
```json
{
  "segment_id":"CN_SH_000123",
  "timestamp":"2025-11-05T19:52:55Z",
  "speed":52.3,
  "flow":1200,
  "occupancy":35.0,
  "quality":{"missing_rate":0.02,"anomaly_score":0.01,"last_updated":"2025-11-05T19:52:50Z"}
}
```

示例:LLM解释请求与响应

请求体(LLM解释):
```json
{
  "predictions":{"speed_forecast":[50.2,48.1,46.5],"density_forecast":[12.5,13.2,14.1]},
  "explain":{"c":8.5,"S":"节点{12-18}网格区域","T":35,"P":0.72},
  "context":{"incident":{"type":"accident","lanes_blocked":2},"weather":{"temp":15.2,"precip":0.0}},
  "style":{"tone":"专业","length":"中"},
  "language":"zh"
}
```

响应体(LLM解释):
```json
{
  "summary":"事故导致上游速度下降,密度上升,形成后向冲击波。",
  "propagation":"预计未来15–30分钟向后传播至节点12–18网格区域。",
  "impact":"影响范围覆盖主干道瓶颈段,峰值延误约8–12分钟。",
  "recommendations":"建议上游提前诱导,调整信号绿波带,开放应急车道。",
  "confidence":0.81
}
```

示例:系统管理(模型注册与审计)

请求体(模型注册):
```json
{
  "model_id":"gcntcn_v1",
  "version":"2025.11",
  "input_schema":{"incident":"GeoJSON/segment_id","traffic_history":"array"},
  "output_schema":{"predictions":"array","explain":"object"},
  "training_data_range":"2025-01-01/2025-10-31",
  "metrics":{"MAE":2.1,"RMSE":3.4,"MAPE":0.055,"R2":0.91},
  "latency_profile":"P1",
  "resource_quota":{"cpu":"4","memory":"8Gi","concurrency":100}
}
```

响应体(模型注册):
```json
{"status":"registered","model_id":"gcntcn_v1","version":"2025.11"}
```

审计事件记录:
```json
{
  "actor":"admin@tenantA",
  "action":"model_register",
  "resource":"gcntcn_v1:2025.11",
  "before":null,
  "after":{"status":"registered"},
  "timestamp":"2025-11-05T19:52:55Z"
}
```

---

## 参考文献

[^1]: 交通速度数据集(METR-LA、PEMS-BAY、PEMSD7)。https://zhuanlan.zhihu.com/p/1932002363905933394  
[^2]: CSDN:METR-LA数据集介绍。https://blog.csdn.net/qq_44858786/article/details/134788448  
[^3]: GitHub - STGCN_IJCAI-18。https://github.com/VeritasYin/STGCN_IJCAI-18  
[^4]: GitHub - T-GCN: Temporal Graph Convolutional Network。https://github.com/lehaifeng/T-GCN  
[^5]: Keras示例:使用图神经网络和LSTM进行交通预测。https://keras.org.cn/examples/timeseries/timeseries_traffic_forecasting/  
[^6]: 知乎专栏:交通流量预测(三)--GraphWaveNet(IJCAI-19)。https://zhuanlan.zhihu.com/p/698262512  
[^7]: 知乎专栏:DGCN论文精读。https://zhuanlan.zhihu.com/p/663016070  
[^8]: AST-DGCN模型深度解析(Scientific Reports 2025解读)。https://www.xugj520.cn/archives/ast-dgcn-traffic-prediction-2.html  
[^9]: HansPub:基于DGCN的交通流量预测。https://pdf.hanspub.org/aam2025142_32624301.pdf  
[^10]: Waze. Live Map & Traffic Events. https://www.waze.com/live-map  
[^11]: X技术网. 考虑因果推断和动态路径诱导相结合的交通管理方法。https://www.xjishu.com/zhuanli/57/202510070381.html  
[^12]: MDPI. Dynamic Route Planning Strategy for Emergency Vehicles Based on Government–Enterprise Collaboration. https://www.mdpi.com/2076-3417/15/21/11496  
[^13]: 百度百科. 车流冲击波。https://baike.baidu.com/item/%E8%BD%A6%E6%B5%81%E5%86%B2%E5%87%BB%E6%B3%A2/22655133  
[^14]: 北京建筑大学学报. CTM模型突发事件下的拥堵传播规律。https://journal.bjut.edu.cn/bjgydxxb/cn/article/pdf/preview/10.11936/bjutxb2015010018.pdf  
[^15]: 中国智能交通网. 基于矩形法的交通拥堵传播模型研究。https://www.chinautc.com/upload/accessorychinautc/201810/2018103191348532954.pdf  
[^16]: 系统工程. 基于时空关联规则挖掘的城市交通拥堵传导预测。https://sysengi.cjoe.ac.cn/CN/10.12011/SETP2020-2752

---

## 结语

本规范以统一的数据口径、明确的接口契约、稳定的版本策略与完善的安全认证为基础,结合物理约束与数据驱动的时空建模方法,提供可解释的拥堵预测与路径规划能力。工程落地建议以GCN+LSTM为稳健基线,根据场景与序列长度引入TCN/Transformer与动态图方法;在应急管理中,滚动时域重规划与因果诱导可显著提升系统韧性与公众满意度。后续工作重点在于跨城基准的统一、应急策略的量化评估与复杂网络下瓶颈耦合机制的在线学习与预警能力建设。