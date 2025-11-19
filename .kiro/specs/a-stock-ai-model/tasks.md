# Implementation Plan

- [x] 1. 搭建项目基础架构和数据库连接



  - 创建Python项目结构（src目录、配置文件、依赖管理）
  - 配置MySQL数据库连接（使用SQLAlchemy ORM）
  - 创建数据库模型类映射现有表（stock_basic_info, stock_kline_data）
  - 实现数据库连接池和会话管理
  - _Requirements: 5.1, 5.2_

- [x] 2. 实现数据加载和预处理模块

  - [x] 2.1 创建StockDataLoader类


    - 实现从stock_kline_data表加载K线数据的方法
    - 实现从stock_basic_info表加载股票信息的方法
    - 实现获取所有活跃股票列表的方法
    - 实现数据完整性验证逻辑
    - _Requirements: 1.1, 1.2, 1.4_
  

  - [x] 2.2 实现数据预处理功能

    - 编写数据清洗函数（处理缺失值、异常值）
    - 实现数据标准化和归一化方法
    - 创建时间序列滑动窗口生成器（用于模型输入）
    - _Requirements: 1.4, 2.2_

- [x] 3. 开发特征工程模块

  - [x] 3.1 实现FeatureEngineer类


    - 编写技术指标计算函数（MA、EMA、MACD、RSI、布林带）
    - 实现价格特征生成（涨跌幅、振幅、价格变化率）
    - 实现成交量特征生成（量比、换手率相关特征）
    - _Requirements: 6.1, 6.5_
  

  - [x] 3.2 创建特征数据集构建器

    - 实现特征矩阵生成逻辑
    - 编写训练集、验证集、测试集划分代码
    - 实现特征缓存机制（避免重复计算）
    - _Requirements: 2.2, 6.1_

- [x] 4. 构建深度学习模型架构


  - [x] 4.1 实现LSTM模型


    - 使用PyTorch定义LSTM网络结构
    - 实现前向传播逻辑
    - 配置模型超参数（层数、隐藏单元数、dropout）
    - _Requirements: 2.1_
  
  - [x] 4.2 实现Transformer模型


    - 使用PyTorch定义Transformer编码器结构
    - 实现位置编码和注意力机制
    - 配置模型超参数（头数、层数、维度）
    - _Requirements: 2.1_
  
  - [x] 4.3 实现GRU模型


    - 使用PyTorch定义GRU网络结构
    - 实现前向传播逻辑
    - 配置模型超参数
    - _Requirements: 2.1_

- [x] 5. 开发模型训练服务

  - [x] 5.1 创建ModelTrainer类


    - 实现训练循环（前向传播、损失计算、反向传播）
    - 实现验证逻辑和早停机制
    - 编写模型检查点保存功能（每10个epoch）
    - 实现学习率调度器
    - _Requirements: 2.3, 2.5_
  


  - [x] 5.2 实现模型评估器



    - 编写评估指标计算函数（MAE、RMSE、MAPE、方向准确率）
    - 实现模型性能可视化（损失曲线、预测对比图）
    - 创建评估报告生成器
    - _Requirements: 2.4_
  

  - [x] 5.3 集成MinIO模型存储

    - 配置MinIO客户端连接
    - 实现模型文件上传和下载功能
    - 实现模型版本管理逻辑
    - _Requirements: 5.1, 5.2_
  

  - [x] 5.4 创建数据库表和ORM模型



    - 创建ai_models表的SQLAlchemy模型
    - 实现模型元数据的数据库存储逻辑
    - 编写模型查询和更新方法
    - _Requirements: 5.1, 5.3_




- [ ] 6. 实现预测服务
  - [x] 6.1 创建PredictionEngine类

    - 实现模型加载和初始化逻辑
    - 编写单股票预测方法（未来5天）
    - 实现批量预测功能
    - 添加GPU/CPU自动检测和设备管理
    - _Requirements: 3.1, 3.3, 3.4_
  

  - [x] 6.2 实现置信区间计算


    - 使用蒙特卡洛dropout或集成方法生成预测分布
    - 计算95%置信区间
    - 实现趋势方向判断逻辑（上涨/下跌/震荡）
    - _Requirements: 3.2, 3.5_
  
  - [x] 6.3 创建TechnicalIndicatorCalculator类


    - 实现实时技术指标计算（基于最新数据）
    - 编写指标更新逻辑
    - 实现指标结果缓存到technical_indicators表
    - _Requirements: 6.1, 6.2, 6.5_
  

  - [x] 6.4 创建ai_predictions表和ORM模型

    - 创建预测结果表的SQLAlchemy模型
    - 实现预测结果存储逻辑
    - 编写预测历史查询方法
    - _Requirements: 5.1_
  

  - [x] 6.5 集成Redis缓存

    - 配置Redis客户端连接
    - 实现预测结果缓存逻辑（TTL 1小时）
    - 实现技术指标缓存逻辑（TTL 30分钟）
    - 编写缓存失效和更新策略
    - _Requirements: 3.3, 6.2_

- [x] 7. 构建FastAPI后端服务


  - [x] 7.1 创建API路由和端点



    - 实现数据查询API（/api/v1/data/stocks, /api/v1/data/stocks/{symbol}/kline）
    - 实现训练API（/api/v1/training/start, /api/v1/training/status）
    - 实现预测API（/api/v1/prediction/predict, /api/v1/prediction/indicators）
    - 实现模型管理API（/api/v1/models/list, /api/v1/models/{model_id}）
    - _Requirements: 3.1, 4.1, 4.2, 4.3_
  
  - [x] 7.2 实现请求验证和错误处理



    - 使用Pydantic定义请求和响应模型
    - 实现统一错误处理中间件
    - 编写输入参数验证逻辑
    - 实现API限流机制
    - _Requirements: 3.3, 5.3_
  
  - [x] 7.3 集成Celery异步任务队列


    - 配置Celery和Redis broker
    - 将模型训练封装为Celery任务
    - 实现任务状态查询接口
    - 编写任务失败重试逻辑
    - _Requirements: 2.3_
  
  - [x] 7.4 添加API文档和日志




    - 配置FastAPI自动生成的Swagger文档
    - 实现结构化日志记录（JSON格式）
    - 添加请求追踪ID
    - _Requirements: 5.3_

- [ ] 8. 开发前端可视化界面
  - [ ] 8.1 创建React项目结构
    - 初始化React + TypeScript项目
    - 配置路由（React Router）
    - 设置状态管理（Redux或Zustand）
    - 配置API客户端（axios）
    - _Requirements: 4.1, 4.2, 4.3, 4.4_
  
  - [ ] 8.2 实现股票搜索和选择组件
    - 创建股票搜索输入框组件
    - 实现自动补全功能
    - 编写股票列表展示组件
    - 添加筛选功能（按行业、市场）
    - _Requirements: 4.2_
  
  - [ ] 8.3 开发K线图和预测可视化组件
    - 集成ECharts或TradingView库
    - 实现K线图展示（历史数据）
    - 叠加技术指标显示（MA、MACD、RSI、布林带）
    - 绘制预测曲线和置信区间
    - 实现图表交互功能（缩放、tooltip）
    - _Requirements: 4.1, 4.3, 4.4, 6.4_
  
  - [ ] 8.4 创建模型管理界面
    - 实现模型列表展示组件
    - 创建模型训练配置表单
    - 显示训练进度和性能指标
    - 实现模型对比功能（多模型预测对比）
    - _Requirements: 4.3, 4.5_
  
  - [ ] 8.5 实现数据加载和状态管理
    - 编写API调用hooks
    - 实现数据缓存和更新逻辑
    - 添加加载状态和错误处理
    - 优化组件渲染性能
    - _Requirements: 4.4_

- [ ] 9. 系统集成和端到端测试
  - [ ] 9.1 集成所有服务模块
    - 连接数据加载、特征工程、模型训练、预测服务
    - 配置环境变量和配置文件
    - 实现服务健康检查端点
    - _Requirements: 所有需求_
  
  - [ ] 9.2 编写端到端测试脚本
    - 测试完整的训练流程（数据加载→训练→保存）
    - 测试完整的预测流程（加载模型→预测→返回结果）
    - 测试前后端集成（API调用→数据展示）
    - _Requirements: 所有需求_
  
  - [ ] 9.3 性能测试和优化
    - 使用Locust进行API压力测试
    - 测试预测服务响应时间（目标<5秒）
    - 优化数据库查询性能
    - 优化模型推理速度
    - _Requirements: 3.3, 5.3_

- [ ] 10. 部署配置和文档
  - [ ] 10.1 创建Docker容器化配置
    - 编写Dockerfile（后端服务、Celery worker）
    - 创建docker-compose.yml（包含MySQL、Redis、MinIO）
    - 配置环境变量和secrets管理
    - _Requirements: 5.2_
  
  - [ ] 10.2 编写部署和运维文档
    - 创建README.md（项目介绍、快速开始）
    - 编写安装和配置指南
    - 文档化API接口
    - 添加常见问题解答
    - _Requirements: 所有需求_
