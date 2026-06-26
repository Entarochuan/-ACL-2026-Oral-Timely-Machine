# 分布式工具服务器

分布式Master-Worker架构的工具服务器实现，支持工具在不同机器上分布式部署和执行。

## 📁 目录结构

```
tool_server/
├── __init__.py          # 模块初始化，导出主要类和函数
├── __main__.py          # 主入口，支持 python -m 运行
├── cli.py               # 命令行界面，支持三种运行模式
├── models.py            # 数据模型定义
├── utils.py             # 通用工具函数
├── master_server.py     # Master服务器实现
├── worker_server.py     # Worker服务器实现
└── README.md           # 说明文档
```

## 🚀 运行模式

### 1. Master模式（推荐）
启动Master服务器，等待Worker注册：
```bash
python -m Bootcampv2.utils.tool_server.cli \
    --mode master \
    --tools_yaml_path config.yaml \
    --port 8000
```

### 2. Worker模式（推荐）

- Worker DLC提交模板: [dlc_launch_bootcamp_tool_server.sh](/Bootcampv2/utils/tool_server/dlc_launch_bootcamp_tool_server.sh)

启动多个Worker服务器并注册到Master：
```bash
python -m Bootcampv2.utils.tool_server.cli \
    --mode worker \
    --tools_yaml_path config.yaml \
    --master_url http://master_ip:master_port \
    --port 8001 \
    --num_workers 3
```

### 3. Unified模式（建议Debug使用）
在单机上同时启动Master和多个Worker：
```bash
python -m Bootcampv2.utils.tool_server.cli \
    --mode unified \
    --tools_yaml_path config.yaml \
    --port 8000 \
    --num_workers 5 \
    --keep_running
```

## 📋 主要参数

- `--mode`: 运行模式 (master/worker/unified)
- `--tools_yaml_path`: 工具配置文件路径
- `--port`: 服务器端口号
- `--num_workers`: Worker服务器数量 (worker和unified模式)
- `--master_url`: Master服务器URL (worker模式必需)
- `--keep_running`: 保持服务器运行 (unified模式)
- `--test_servers`: 启动后测试服务器功能
- `--log_dir`: 日志目录路径

## 🔧 代码使用

```python
from Bootcampv2.utils.tool_server import DistributedMasterServer, DistributedWorkerServer

# 创建Master服务器
master = DistributedMasterServer(tools_config, "0.0.0.0", 8000)
master.run()

# 创建Worker服务器
worker = DistributedWorkerServer(tools_config, "0.0.0.0", 8001, "worker_1", "http://master:8000")
worker.run()
```

## ✨ 主要特性

1. **动态Worker注册**: Worker可以动态注册到Master，支持热插拔
2. **负载均衡**: Master自动将请求分配到负载最轻的Worker
3. **健康监控**: Master监控Worker健康状态，自动清理死掉的Worker
4. **实例映射**: 支持instance_id到Worker的路由映射
5. **多Worker支持**: 支持启动多个Worker实例提高并发能力
6. **配置自动生成**: 自动生成适配分布式架构的配置文件

## 🔄 工作流程

1. **Master启动**: 启动Master服务器，等待Worker注册
2. **Worker注册**: Worker启动后自动注册到Master
3. **请求路由**: Master接收请求，根据负载均衡选择Worker
4. **实例跟踪**: 创建实例时建立映射，后续请求路由到同一Worker
5. **健康监控**: Master定期检查Worker健康状态
6. **自动清理**: 检测到Worker死亡时自动清理相关映射

## 📊 监控与测试

使用 `--test_servers` 参数可以在启动后自动测试服务器功能：
```bash
python -m Bootcampv2.utils.tool_server.cli \
    --mode unified \
    --tools_yaml_path config.yaml \
    --port 8000 \
    --num_workers 3 \
    --test_servers \
    --keep_running
```

访问 `http://server:port/health` 查看服务器状态和Worker信息。 