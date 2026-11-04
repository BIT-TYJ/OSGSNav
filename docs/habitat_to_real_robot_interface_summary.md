# IEVE Habitat 接口与实机替换汇总

## 1. 文档范围

本文汇总 IEVE 默认运行链中直接使用的 Habitat 接口，并给出对应的实机替代输入、输出和处理方式。

默认运行链为：

```text
main.py
  -> envs.make_vec_envs()
  -> Habitat VectorEnv
  -> Instance_Exp_Env_Agent
  -> imagegoal_env.ObjectGoal_Env
  -> habitat.RLEnv
```

本文重点覆盖默认 `--agent sem_exp` 路径。仓库中 `t_dis_env.py`、`t_class_env.py` 和旧 `objectgoal_env.py` 的额外 Habitat 接口单独列在第 9 节。`3rdparty/habitat-lab` 内部实现不属于 IEVE 实机后端的直接接口，不逐项展开。

## 2. 总体替换原则

Habitat 当前同时提供图像、深度、语义真值、机器人真值位姿、episode 目标、碰撞结果和评测指标。实机部署时，需要将这些能力拆分到真实传感器、定位系统、任务管理器、控制器和安全层：

```text
Habitat RGB-D                 -> 实机 RGB-D 相机
Habitat semantic truth        -> Mask R-CNN 等视觉模型
Habitat agent ground truth    -> Point-LIO/FAST-LIO/其他定位系统
Habitat episode dataset       -> 实机 EpisodeGoal 任务接口
Habitat collision/occupancy   -> LiDAR 占据图、bumper 和安全层
Habitat velocity action       -> STG 闭环控制器和 /cmd_vel
Habitat success/SPL           -> 实机独立评测和人工真值
```

算法内部可以继续保留目标匹配、语义建图、IEVE/MinAV 状态管理、FMM 规划和 3DGS，但它们消费的数据必须具有明确的时间戳、单位和坐标系。

## 3. Habitat 观测输入接口

| Habitat 接口或字段 | 当前用途 | 实机替代输入 | 处理要求 |
|---|---|---|---|
| `obs["rgb"]` | 检测、分割、特征匹配、3DGS | RGB-D 相机彩色图 | `uint8[H,W,3]`，统一为 RGB |
| `obs["depth"]` | 二维建图、目标投影、3DGS | 对齐到彩色相机的深度图 | 入口统一转为 `float32[H,W]` 米，`0` 表示无效 |
| `obs["semantic"]` | Habitat 实例级语义真值 | Mask R-CNN 或其他实机分割结果 | 删除所有仿真语义真值依赖 |
| `obs["instance_imagegoal"]` | 目标实例参考图 | 文件、ROS service/action 或上位机输入的目标图 | 每个 episode 必填 |
| `obs["instance_imagegoal_hfov"]` | 目标图相机水平视场角 | 目标图内参或元数据 | 当前主代码未直接读取，仍应明确是否提供 |
| `obs["gps"]` | Habitat GPS lab sensor | LIO/SLAM 输出的平移 | 配置存在，但当前代码实际读取 simulator 真值位姿 |
| `obs["compass"]` | Habitat Compass lab sensor | LIO/IMU 输出的航向 | 配置存在，但当前代码实际读取 simulator 真值旋转 |
| RGB/Depth `width,height,hfov` | 构造图像和相机模型 | `CameraInfo.width/height/K` | 使用真实内参，不再只由 `hfov` 推导 |
| Sensor `position` | 仿真相机安装高度 | `T_base_camera` 静态外参 | 从 URDF、TF 或标定文件读取 |
| Depth `min_depth/max_depth` | Habitat 归一化深度恢复 | 相机有效量程和 `depth_scale` | 根据消息编码显式转换，禁止猜测深度单位 |

相关代码位置：

- `envs/habitat/configs/tasks/instance_imagenav.yaml`：RGB、Depth、Semantic sensor 配置。
- `3rdparty/habitat-lab/habitat/config/habitat/task/instance_imagenav_v3.yaml`：目标图、HFOV、GPS、Compass 和 measurement 配置。
- `envs/habitat/imagegoal_env.py::reset/step`：读取 Habitat observation。
- `agents/instance_exp.py::_preprocess_depth`：当前按 Habitat 归一化深度处理。

### 3.1 深度输入特别说明

Habitat Depth 默认可以输出归一化深度，而实机常见编码为：

```text
16UC1 -> 整数深度值，需要乘设备提供的 depth_scale
32FC1 -> 通常为米，但仍需根据驱动契约确认
```

实机适配器入口应统一执行：

```text
NaN/Inf/越界/无效深度 -> 0.0
有效深度             -> float32 meters
```

RGB 和 Depth 必须属于同一时间附近，并且使用对齐到彩色相机的深度。若图像发生 resize 或 crop，必须同步更新 `fx`、`fy`、`cx` 和 `cy`。

## 4. Habitat 位姿与 Simulator 接口

| Habitat 接口 | 返回内容 | 当前用途 | 实机替代 |
|---|---|---|---|
| `sim.get_agent_state(0)` | `position + rotation` 真值 | 当前主要定位来源 | `T_odom_base(t)` 和 `T_odom_camera_optical(t)` |
| `get_sim_location()` | `[x,y,yaw]` | 二维地图、FMM、移动量计算 | `base_pose_2d=[x,y,yaw]`，单位米和弧度 |
| `get_pose_change()` | `[dx,dy,dyaw]` | `info["sensor_pose"]` | 从相邻 `T_odom_base` 计算 |
| `get_transformation_matrix()` | 相对起点的 4x4 变换 | 轨迹和相机变换 | 基于带时间戳的 LIO/SLAM 位姿计算 |
| `sim.config.sim_cfg.scene_id` | Habitat 场景 ID | 日志和 episode 数据加载 | `site_id`、`session_id` 或 `map_id` |
| agent `position[1]` | 仿真高度 | 动态 `agent_height` | 相机外参或真实三维位姿 |
| `ieve_gt_pose` | Habitat 二维绝对真值 | Semantic Mapping 绝对位姿 | 实机 `base_pose_2d` |
| `agent_position` | Habitat 三维位置 | MinAV、占据图和部分 3DGS 逻辑 | 拆成 `base_position` 与 `camera_position` |
| `agent_rotation_xyzw` | Habitat 四元数 | MinAV、目标投影和 3DGS | 从 `T_odom_camera_optical` 提取 |
| `camera_position` | agent 位置加相机高度 | 3DGS | 直接从 TF 链计算相机位姿 |

实机必须区分以下两个位姿：

```text
T_odom_base(t_cam)             # 二维地图、FMM、底盘控制和移动进度
T_odom_camera_optical(t_cam)   # RGB-D 反投影、目标定位、匹配和 3DGS
```

相机位姿必须对应相机曝光时间 `t_cam`，不能直接使用 ROS callback 到达时间。建议维护 LIO pose buffer：平移使用线性插值，旋转使用 quaternion SLERP。无法插值、TF 缺失或位姿过期时，不得消费该帧，并应触发停车。

## 5. Habitat Episode 与目标接口

| Habitat 接口或字段 | 当前用途 | 实机替代 |
|---|---|---|
| `habitat.RLEnv.reset()` | 切换 episode 并生成首帧 | 停车、检查传感器、接收目标、记录起点和清空算法状态 |
| `current_episode.episode_id` | episode 标识 | 自定义 `episode_id` |
| `current_episode.goal_object_id` | 目标实例真值 ID | 算法中删除；仅可作为人工评测标注 |
| `current_episode.start_position` | 仿真起点 | episode 开始时的 `T_odom_base` |
| `current_episode.start_rotation` | 仿真起始姿态 | episode 开始时的真实姿态 |
| dataset `object_category` | 目标类别真值 | 操作者指定，或从目标图检测/分类 |
| dataset `goal_object_id` | Habitat 对象 ID | 删除算法依赖 |
| dataset `geodesic_distance` | 起点到目标的最短路径真值 | 有预建真值地图时离线计算，否则不提供 |
| dataset `euclidean_distance` | 起点到目标的欧氏距离真值 | 人工标注或外部定位评测系统 |
| dataset scene/episode slice | 并行仿真评测 | 实机模式禁用 |

建议新增实机目标结构：

```text
EpisodeGoal:
  episode_id
  target_rgb
  target_category_name
  target_category_id
  timeout_s
  operator_start_time
```

实机模式不能使用 `--goal_cat_source oracle`。优先由任务接口明确提供目标类别；无法提供时，再使用目标图 detector/classifier 的预测结果。

## 6. Habitat 动作接口

| Habitat 接口或字段 | 当前含义 | 实机替代 |
|---|---|---|
| `super().step(action)` | 执行仿真动作并返回下一帧 | 发布 STG/速度命令，并等待真实的新观测和执行反馈 |
| `action=("velocity_control","velocity_stop")` | Habitat 组合动作 | 实机控制状态机 |
| `linear_velocity` | Habitat 归一化线速度 | `linear_x_mps` |
| `angular_velocity` | Habitat 归一化角速度 | `angular_z_radps` |
| `camera_pitch_velocity` | 仿真相机俯仰速度 | 固定相机时删除；云台相机时接云台控制器 |
| `velocity_stop` | Habitat terminal STOP | 锁存零速度并结束当前实机 episode |
| `time_step` | 仿真精确动作持续时间 | 基于真实位姿闭环执行，不依赖理论持续时间 |
| `rew` | Habitat reward | 实机调试可返回 `0`，评测指标单独记录 |

当前 Habitat velocity controller 会对归一化输入做尺度映射，因此代码中的速度数组不能直接发布为 ROS `Twist`。实机建议使用以下链路：

```text
IEVE/MinAV goal
  -> FMM short-term goal (STG)
  -> 基于 T_odom_base 的闭环控制器
  -> 物理单位 v/omega
  -> 独立安全层
  -> /cmd_vel
```

控制器应以物理单位工作，并具备速度、角速度、加速度和命令超时限制。

## 7. Habitat 指标、结束与碰撞接口

| Habitat 接口或 measurement | 当前含义 | 实机替代 |
|---|---|---|
| `get_metrics()["distance_to_goal"]` | Habitat 测地距离 | 算法估计距离或外部真值距离 |
| `get_metrics()["success"]` | Habitat success | 算法判断和人工/外部真值判断分别记录 |
| `get_metrics()["spl"]` | Success weighted by Path Length | 有真值地图时离线计算，否则禁用 |
| `get_metrics()["soft_spl"]` | 软成功路径指标 | 有真值时离线计算，否则禁用 |
| `episode_over` | Habitat episode 结束状态 | 完成、超时、人工停止、安全故障或定位丢失 |
| `collisions` | 仿真碰撞 measurement | LiDAR 安全区、bumper 和底盘故障反馈 |
| `num_steps` | 仿真步数 | 控制周期数，并额外记录真实耗时 |
| `distance_to_goal_reward` | 距离奖励 | 实机调试可设为 `0`，训练或评测另行定义 |
| 位姿不动碰撞推断 | 发前进命令但位置未变化 | 仅保留为 stuck 检测，不能代替主动避障 |

实机 `done` 至少应覆盖：

```text
目标导航完成
任务超时
人工停止
安全层触发
定位丢失或位姿超时
RGB-D/LiDAR/TF 数据超时
控制器或底盘故障
```

## 8. Habitat 环境与 VectorEnv 接口

| Habitat 接口 | 当前用途 | 实机处理 |
|---|---|---|
| `habitat.config.default.get_config()` | 读取 Habitat 配置 | real backend 改读实机 YAML |
| `habitat.make_dataset()` | 创建 HM3D episode dataset | real backend 禁用 |
| `dataset.episodes` | 场景和 episode 管理 | 替换为实机任务队列 |
| `habitat.RLEnv` 继承 | 将规划逻辑与 Habitat 生命周期绑定 | 将规划核心与 backend 解耦 |
| `env.seed()` | 仿真随机种子 | 实机环境无需使用 |
| `VectorEnv` | 多进程并行仿真 | 第一版换为单进程 `SingleEnvAdapter` |
| `reset()` | 初始化仿真 episode | 保留上层签名，改为实机 episode 初始化 |
| `step()/plan_act_and_preprocess()` | 规划并推进仿真 | 保留兼容签名，内部调用 real backend |
| `observation_space/action_space` | Gym 接口描述 | 提供兼容属性或实机自定义空间 |
| `current_episode` | Habitat 当前任务 | 实机当前 `EpisodeGoal` |
| `get_metrics()` | Habitat 指标 | 实机指标采集器 |
| `episode_over` | Habitat 结束标志 | 实机 episode 状态机 |
| `close()` | 关闭仿真 worker | 锁存停车并关闭 subscriber/controller |

第一版不建议在 Habitat 的多进程 daemon worker 中直接启动 ROS executor 和 CUDA 推理。建议 real backend 强制 `num_processes=1`，同时保持观测张量首维为 1，以减少 `main.py` 的改动。

## 9. 非默认与遗留 Habitat 接口

以下接口位于 `t_dis_env.py`、`t_class_env.py`、旧 `objectgoal_env.py` 等非默认路径。实机迁移时应隔离这些模块，避免意外调用：

| 遗留 Habitat 接口 | 作用 | 实机处理 |
|---|---|---|
| `sim.set_agent_state()` | 瞬移仿真机器人 | 实机不支持，删除 |
| `sim.get_observations_at()` | 在任意位姿渲染观测 | 实机不支持，删除或改为历史数据查询 |
| `sim.semantic_annotations()` | 获取场景语义真值 | 删除 |
| `sim.pathfinder.get_random_navigable_point()` | 采样仿真可达点 | 换成真实占据图的可达点采样 |
| `sim.get_sensor_observations()` | 获取 simulator 原始传感器输出 | 换成实机传感器缓存 |
| `sim._sensor_suite.get_observations()` | 将 simulator 数据转换为 Habitat observation | 删除私有 Habitat 接口依赖 |
| `sim_map_to_sim_continuous()` | 仿真地图与世界坐标转换 | 换成 grid 与 `odom` 坐标转换 |
| Habitat scene teleport | 重置或采样仿真位姿 | 实机完全禁止 |
| `habitat_sim.utils.common.d3_40_colors_rgb` | 仿真语义可视化颜色 | 仅可视化依赖，可自行替换 |

## 10. 建议的实机输入输出契约

### 10.1 每帧实机输入

| 字段 | 类型或形状 | 单位/坐标系 | 用途 |
|---|---|---|---|
| `raw_rgb` | `uint8[H,W,3]` | RGB | 检测、匹配、3DGS |
| `raw_depth_m` | `float32[H,W]` | 米，`0` 无效 | 建图、投影、3DGS |
| `camera_K` | `float64[3,3]` | 当前图像分辨率 | 反投影和目标方位计算 |
| `camera_distortion` | array + model | 相机模型 | 未校正图像使用 |
| `stamp` | ROS time | 相机曝光时间 | 数据同步基准 |
| `frame_id/sequence` | string/int | - | 去重和日志追踪 |
| `T_odom_base` | `float64[4,4]` | ROS REP-103 | 二维规划和控制 |
| `T_odom_camera_optical` | `float64[4,4]` | ROS optical frame | RGB-D 和 3DGS 位姿 |
| `base_pose_2d` | `[x,y,yaw]` | 米、弧度，`odom` | 替换 `ieve_gt_pose` |
| `sensor_pose` | `[dx,dy,dyaw]` | 米、弧度 | 兼容当前接口和诊断 |
| `nav_occupancy` | `uint8[M,M]` | IEVE 栅格坐标 | LiDAR 导航障碍层 |
| `visual_explored` | `uint8[M,M]` | IEVE 栅格坐标 | 相机真正观察过的区域 |
| `linear_velocity_feedback` | float | m/s | 停稳和执行状态判断 |
| `angular_velocity_feedback` | float | rad/s | 停稳和执行状态判断 |
| `pose_age_ms` | float | ms | 定位超时保护 |
| `frame_age_ms` | float | ms | 图像超时保护 |
| `localization_health` | enum/bool | - | 定位故障停车 |
| `safety_state` | enum | - | 急停、障碍和控制授权 |

`nav_occupancy` 与 `visual_explored` 必须分开。Mid-360 可以观察机器人四周的自由空间，但不能证明相机已经在这些区域完成目标视觉搜索。

### 10.2 实机动作输出

```text
MotionCommand:
  linear_x_mps
  angular_z_radps
  duration_s
  terminal_stop
  source
  stamp
```

### 10.3 实机 step 兼容返回值

为减少上层代码改动，`RealRobotEnv` 可以暂时保持以下接口：

```text
reset() -> obs, info
step(command) -> obs, reward, done, info
plan_act_and_preprocess(planner_inputs) -> obs, reward, done, info
```

其中：

- `obs` 保持首维为 1 的现有张量约定。
- `reward` 在实机调试阶段可以固定为 `0.0`。
- `done` 由实机 episode 状态机生成。
- `info` 不再包含伪造的 Habitat SPL 或真值距离。

## 11. 推荐实施顺序

1. 冻结一套当前 Habitat 仿真基线和运行配置。
2. 新增 `habitat/real` backend 选择，保持仿真路径不变。
3. 用 rosbag 接入 RGB、aligned depth、CameraInfo、LIO pose 和 TF，暂不控制底盘。
4. 修正深度单位、真实内参、图像 resize 和坐标系转换。
5. 使用 `T_odom_base` 替换 `ieve_gt_pose`，使用 `T_odom_camera_optical` 接入 3DGS。
6. 从 Mid-360 点云生成独立 `nav_occupancy`，不要覆盖 `visual_explored`。
7. 在 shadow mode 下运行完整 IEVE/MinAV/FMM 链路，只记录 STG 和速度建议。
8. 接入低速闭环控制器和独立安全层，再开始停走式实机运动。
9. 最后建立人工真值或外部定位评测，单独计算实机成功率和路径指标。
