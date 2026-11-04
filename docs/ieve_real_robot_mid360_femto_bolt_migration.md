# IEVE 从 Habitat 迁移到移动机器人实机的修改清单与实施顺序

本文档针对当前仓库 `/nas_ds925/tyj/IEVE`，硬件和工具约束如下：

- 定位传感器：DJI Livox Mid-360（点云 + 内置 IMU）。
- RGB-D 相机：Orbbec Femto Bolt。
- LiDAR-相机外参：`direct_visual_lidar_calibration`。
- 推荐中间件：ROS 2；以下使用 ROS 标准 TF/话题概念，不把具体话题名写死。

本文只描述迁移设计、代码接口和落地顺序，不代表这些实机接口已经在仓库中实现。

## 1. 结论

不要把 Femto Bolt 图像直接塞进现有 Habitat 环境后就发送速度。当前代码不只在环境创建处依赖 Habitat，还依赖了 Habitat 的深度编码、真值位姿、相机坐标系、速度归一化、episode 数据和评测指标。

建议采用以下结构：

```text
Mid-360 point cloud + IMU -> FAST-LIO2 -> T_odom_lidar(t)
                                              |
Femto Bolt RGB-D + CameraInfo ----------------+-> 时间对齐/TF查询
                                              |
                                   RealRobotEnv observation
                                              |
                            IEVE semantic map / MinAV / 3DGS
                                              |
                              physical velocity command (m/s, rad/s)
                                              |
                              LiDAR safety filter -> base controller
```

第一版建议使用 `livox_ros_driver2 + FAST-LIO2`。`direct_visual_lidar_calibration` 只负责 LiDAR 与相机的空间外参，不负责持续定位，也不能替代时间同步、`base_link` 外参或底盘安全控制。

总体原则是：新增实机后端，保留 Habitat 后端用于回归；上层 IEVE 接收统一的、带单位和坐标系声明的数据契约。

## 2. 当前代码中 Habitat 的耦合位置

当前主链路为：

```text
main.py
  -> envs.make_vec_envs()
  -> envs/habitat.construct_envs()
  -> Instance_Exp_Env_Agent(ObjectGoal_Env)
  -> Habitat reset/step
  -> RGBD + simulator pose + simulator metric
  -> semantic mapping / FMM / MinAV / 3DGS
```

需要处理的耦合点如下。

| 位置 | 当前行为 | 实机迁移要求 |
|---|---|---|
| [`envs/__init__.py`](../envs/__init__.py#L3) | 无条件导入并创建 Habitat vector env | 增加 `--env_backend habitat/real` 分派；实机固定 `num_processes=1` |
| [`envs/habitat/__init__.py`](../envs/habitat/__init__.py#L72) | 创建 Habitat dataset、scene、GPU simulator 和传感器 | 保持为纯仿真后端；不要在这里堆 ROS 分支 |
| [`envs/habitat/imagegoal_env.py`](../envs/habitat/imagegoal_env.py#L26) | `ObjectGoal_Env` 直接继承 `habitat.RLEnv` | 抽取后端无关的数据契约/episode 状态，新增 `RealRobotEnv` |
| [`agents/instance_exp.py`](../agents/instance_exp.py#L19) | 规划和感知类直接继承 Habitat `ObjectGoal_Env` | 将感知/规划核心与传感器、动作执行后端解耦，避免复制整份 4000 行文件 |
| [`main.py`](../main.py#L471) | `make_vec_envs()` 后按 Habitat episode 运行 | 实机改为单环境、外部触发 episode、可中止的循环 |
| [`main.py`](../main.py#L140) | 使用名为 `ieve_gt_pose` 的仿真真值绝对位姿 | 改名为 `base_pose_2d`；兼容期可回退读取旧字段 |
| [`envs/habitat/imagegoal_env.py`](../envs/habitat/imagegoal_env.py#L571) | 从 simulator state 计算平面位姿和增量 | 实机从曝光时刻的 `T_odom_base` 计算 |
| [`envs/habitat/imagegoal_env.py`](../envs/habitat/imagegoal_env.py#L596) | RGB、归一化深度、agent pose 全来自 Habitat | 实机适配器生成同等语义但显式带单位/时间戳的数据 |
| [`main.py`](../main.py#L102) | 用 `hfov` 假设 `fx=fy`、主点在图像中心 | 从 Femto Bolt `CameraInfo.K` 读取真实内参 |
| [`agents/instance_exp.py`](../agents/instance_exp.py#L3584) | 假设深度为 Habitat `[0,1]` 归一化值 | 改为显式 depth encoding；实机统一为米 |
| [`model.py`](../model.py#L151) | 语义建图仅使用 `hfov`、相机高度和零俯仰 | 使用真实 `K` 和 `T_base_camera_optical` |
| [`envs/utils/depth_utils.py`](../envs/utils/depth_utils.py#L394) | MinAV 全图投影用固定 optical flip + camera height | 传入真实 `K` 和 `T_world_camera_optical` |
| [`main.py`](../main.py#L212) | 3DGS 使用 `camera_position`，但旋转仍取 `agent_rotation_xyzw` | 传完整相机位姿矩阵，不再拼接车体姿态 |
| [`gaussian_mapping/mapper/realtime_3dgs.py`](../gaussian_mapping/mapper/realtime_3dgs.py#L457) | 隐式添加 OpenCV/OpenGL 翻转 | 新增显式 `T_world_camera_optical`/`T_world_camera_gl` 转换并测试 |
| [`agents/instance_exp.py`](../agents/instance_exp.py#L2089) | 输出 Habitat 归一化速度和可变 `time_step` | 改成物理单位 `v_mps/w_radps`，经安全层发布 |
| [`envs/habitat/imagegoal_env.py`](../envs/habitat/imagegoal_env.py#L543) | 读取 Habitat success/SPL/distance | 实机建立独立指标，不能伪装成 Habitat 指标 |

## 3. 推荐新增的模块

建议新增以下文件，而不是把 ROS 回调写入 `envs/habitat/*`：

```text
envs/real_robot/
  __init__.py
  contracts.py          # FrameObservation、MotionCommand、EpisodeGoal
  sensor_buffer.py      # RGB/depth/CameraInfo/pose 时间配对和位姿插值
  transforms.py         # ROS -> IEVE、base -> camera、四元数/矩阵转换
  goal_provider.py      # 目标图加载、开始/停止 episode 接口
  ros_env.py            # reset/step/plan_act_and_preprocess 的实机实现
  single_env_adapter.py # 保持 main.py 所需的批量维度，num_envs=1
  safety_controller.py  # 限速、加速度限制、超时停车、LiDAR 碰撞保护

configs/real_robot/
  mid360_femto_bolt.yaml

tests/real_robot/
  test_depth_contract.py
  test_intrinsics_resize.py
  test_frame_transforms.py
  test_motion_command.py
  test_recorded_frame_contract.py
```

实机不要复用当前多进程 `VectorEnv` 的 daemon worker。ROS executor、CUDA 和 `fork` 混用会增加退出、信号和显存问题。第一版使用同进程 `SingleEnvAdapter`，但保持返回数据首维为 1，使 `main.py` 的张量逻辑少改。

## 4. 实机输入输出契约

### 4.1 每帧观测

`RealRobotEnv` 应在同一个相机时间戳 `t_cam` 上组装以下字段：

| 字段 | 类型/形状 | 单位/坐标系 | 用途 |
|---|---|---|---|
| `raw_rgb` | `uint8[H,W,3]` | RGB | Mask R-CNN、匹配、3DGS |
| `raw_depth_m` | `float32[H,W]` | 米，0 表示无效 | 语义地图、目标投影、3DGS |
| `camera_K` | `float64[3,3]` | 对应当前 RGB/对齐深度分辨率 | 全部反投影 |
| `camera_distortion` | 数组 + model | 对应实际图像 | 仅未校正图像使用；建议运行时输入已校正图 |
| `stamp` | ROS time | 相机曝光时间 | 同步基准 |
| `T_odom_base` | `float64[4,4]` | ROS REP-103 | 机器人二维规划位姿 |
| `T_odom_camera_optical` | `float64[4,4]` | ROS optical frame | RGB-D/3DGS 的相机位姿 |
| `base_pose_2d` | `[x,y,yaw]` | 米、弧度，`odom` | 替换 `ieve_gt_pose` |
| `sensor_pose` | `[dx,dy,dyaw]` | base 上一帧到当前帧 | 兼容现有接口和诊断 |
| `nav_occupancy` | `uint8[M,M]` | 与 IEVE 栅格对齐 | Mid-360 障碍/可通行性 |
| `visual_explored` | `uint8[M,M]` | 与 IEVE 栅格对齐 | 相机真正看过的区域 |
| `pose_age_ms` | float | 毫秒 | 超时保护 |
| `frame_id`/`sequence` | string/int | - | 日志追踪、去重 |

注意：`nav_occupancy` 和 `visual_explored` 必须分开。Mid-360 是 360 度传感器，如果把 LiDAR 的自由空间直接写入当前 `exp_pred`，IEVE 会把机器人身后“雷达看过但相机没看过”的区域误判为视觉已探索，从而漏检目标。

### 4.2 目标输入

Habitat 的 `instance_imagegoal` 和 `object_category` 来自 dataset；实机必须新增 `EpisodeGoal`：

```text
episode_id
target_rgb                 # 必填
target_category_name       # 建议提供
target_category_id         # 可选
operator_start_time
timeout_s
```

目标可以从文件、ROS service/action 或上位机界面传入。当前检测器只映射有限的 COCO 类别，类别范围见 [`constants.py`](../constants.py) 和 [`agents/utils/semantic_prediction.py`](../agents/utils/semantic_prediction.py#L42)。实机不能使用 `--goal_cat_source oracle`；优先明确传入类别，无法提供时再使用 detector/classifier。

### 4.3 动作输出

新增物理动作结构，不要复用含义含混的 Habitat dict：

```text
MotionCommand:
  linear_x_mps
  angular_z_radps
  duration_s
  terminal_stop
  source
  stamp
```

当前 Habitat 控制器将 `[-1,1]` 映射到 `[0,5] m/s` 和 `[-5,5] rad/s`，见 [`default_structured_configs.py`](../3rdparty/habitat-lab/habitat/config/default_structured_configs.py#L215)。例如当前代码中的 `linear_velocity=0` 在该控制器中并不是 0 m/s。实机层必须接收物理单位，不能把 [`agents/instance_exp.py`](../agents/instance_exp.py#L2131) 产生的数组直接发布为 `Twist`。

## 5. 各文件具体应该怎么改

### 5.1 环境工厂与配置

修改 [`arguments.py`](../arguments.py#L89)：

- 增加 `--env_backend habitat/real`。
- 增加实机配置文件路径，不要为每个 topic 增加大量命令行参数。
- real 模式强制 `num_processes=1`、`eval=1`、禁止 Habitat episode slice。
- 增加 `control_rate_hz`、`command_timeout_ms`、`max_linear_mps`、`max_angular_radps`、最大加速度、传感器最大延迟等安全参数。
- `env_frame_width/env_frame_height` 必须由实际 RGB 流确定。当前默认值是 `480 x 640`，不要默认认为它等于 Femto Bolt 常用的 `640 x 480`。

修改 [`envs/__init__.py`](../envs/__init__.py#L6)：

```text
habitat -> construct_envs(args) -> VecPyTorch
real    -> RealRobotEnv(args) -> SingleEnvAdapter -> VecPyTorch
```

保持 [`envs/habitat/__init__.py`](../envs/habitat/__init__.py#L106) 只负责仿真，以便继续做回归测试。

### 5.2 将规划核心从 Habitat 继承链中拆出

当前 [`Instance_Exp_Env_Agent`](../agents/instance_exp.py#L37) 继承 [`ObjectGoal_Env`](../envs/habitat/imagegoal_env.py#L26)，其 `reset()` 和 `plan_act_and_preprocess()` 又通过 `super()` 调 simulator。这是实机迁移最大的结构性接口。

推荐拆成组合关系：

```text
InstanceExpAgentCore
  - preprocess_observation(frame)
  - plan(planner_inputs) -> MotionIntent
  - update_episode_state(...)

HabitatRobotBackend
  - reset/observe/execute/get_metrics

RealRobotBackend
  - reset/observe/execute/get_metrics
```

迁移时保留 `_plan()`、FMM、frontier、实例判断和可视化代码；替换以下边界：

- [`reset()`](../agents/instance_exp.py#L125)：从 backend 获取首帧和目标，而非调用 Habitat `super().reset()`。
- [`plan_act_and_preprocess()`](../agents/instance_exp.py#L1100)：规划只产生动作意图，执行由 backend 完成。
- [`super().step(action)`](../agents/instance_exp.py#L1182)：改为 `backend.execute(command)` 后等待一组新的同步观测。
- `_log_minav_stop_debug()` 中对 `habitat_env.get_metrics()` 的读取要改为可选 backend metric。

不要复制一份 `instance_exp_real.py` 后长期双线维护；该文件内 MinAV/frontier 修改很多，复制后很快会分叉。

### 5.3 深度和相机内参

Femto Bolt 驱动输出的深度常见为整数深度单位，具体 scale 必须从驱动/设备元数据读取，不能硬编码“永远是毫米”。在实机适配器入口立即转换成米并执行：

```text
invalid/NaN/Inf/out-of-range -> 0.0
valid depth                 -> float32 meters
```

必须修改以下位置：

1. [`agents/instance_exp.py::_preprocess_depth`](../agents/instance_exp.py#L3584) 当前按 Habitat `[0,1]` 处理，且把无效值设为超大深度。改为显式输入编码，并让送入 `Semantic_Mapping` 的通道 3 明确为厘米。
2. [`main.py::depth_to_3dgs_meters`](../main.py#L123) 当前通过“整幅深度最大值是否小于 1.5”猜测编码。真实场景全部小于 1.5 m 时会被错误二次缩放。改为由 `depth_unit/depth_encoding` 明确选择。
3. [`main.py::build_3dgs_camera_params`](../main.py#L102) 改为读取 `camera_K`。
4. [`envs/utils/depth_utils.py::get_camera_matrix`](../envs/utils/depth_utils.py#L27) 和点云反投影函数改为支持不同的 `fx/fy/cx/cy`。
5. [`model.py::Semantic_Mapping`](../model.py#L175) 使用处理后分辨率对应的真实内参。
6. [`update_minav_full_grid_from_depth`](../envs/utils/depth_utils.py#L394) 接收 `K`，删除内部 `hfov` 推导。

若把 RGB-D 从原始 `(W0,H0)` 缩放为 `(W1,H1)`，必须同步缩放：

```text
fx1 = fx0 * W1 / W0
fy1 = fy0 * H1 / H0
cx1 = cx0 * W1 / W0
cy1 = cy0 * H1 / H0
```

如果还有裁剪，则主点还要减去裁剪偏移。当前 `_preprocess_obs()` 使用步进抽样，见 [`agents/instance_exp.py`](../agents/instance_exp.py#L3572)，实机建议改为明确 resize/nearest-depth 流程并生成对应的新 `K`。

实例目标方位也不能再用线性 `hfov` 近似。将 [`compute_ins_goal_map()`](../agents/instance_exp.py#L627) 等处替换为：

```text
bearing = atan2(u - cx, fx)
```

### 5.4 Femto Bolt RGB-D 接口

运行时应使用“对齐到彩色相机”的深度，并使用彩色相机对应的 `CameraInfo.K`。建议逻辑话题配置为：

```text
color_image
aligned_depth_image
color_camera_info
```

实际 topic 名通过 `ros2 topic list` 和驱动 launch 参数确认，不在算法代码中硬编码。固定运行时分辨率、帧率、深度对齐模式和图像是否已去畸变；外参标定和运行时必须使用完全相同的成像模式。

对每帧检查：

- RGB、aligned depth 的宽高一致。
- `CameraInfo` 的宽高和图像一致。
- 深度在墙角/物体边缘与 RGB 对齐。
- 自动曝光造成的时间戳/曝光中点差异有记录。
- 相机掉线、重复帧、深度空帧会触发不运动状态。

### 5.5 Mid-360 定位接口

推荐先用 `livox_ros_driver2` 保留 Mid-360 每个点的时间信息，再用 FAST-LIO2 做去畸变和激光惯性里程计。必须核对：

- 点云消息格式和单点时间单位。
- Mid-360 内部 LiDAR-IMU 外参方向。
- IMU 频率、噪声、重力初始化和时间戳。
- FAST-LIO2 输出 pose 对应的是 LiDAR、IMU 还是自定义 body frame。

不要看到一个 Odometry topic 就直接当成 `base_link`。通过 TF 明确得到：

```text
T_odom_base(t_cam)
T_odom_camera_optical(t_cam)
```

每次从当前位置开始探索时，可将连续的 LIO `odom` 作为 IEVE 世界原点。若需要预建地图定位，还要另加重定位/`map -> odom`；这不是 `direct_visual_lidar_calibration` 的职责。

当前 3DGS 把历史 keyframe pose 当成固定值。第一版不要在运行中对历史轨迹做大幅回环跳变，否则新旧高斯会错层。先使用连续 `odom`；以后若引入回环，需同步修正历史 keyframe/高斯地图。

### 5.6 使用 direct_visual_lidar_calibration

建议按以下顺序使用该项目：

1. 先固定 Femto Bolt 的分辨率、去畸变方式和内参。
2. 先确认 Mid-360 点云单点时间和 LIO 去畸变正确。
3. 传感器刚性安装后，在静态、纹理和几何边缘丰富的环境采集图像与点云；覆盖不同距离和视角。
4. 给 `direct_visual_lidar_calibration` 提供与运行时完全一致的图像、`CameraInfo` 和点云，并提供合理的外参初值。
5. 保存工具输出的 4x4 外参和元数据：输入 frame、输出 frame、矩阵方向、图像模式、时间、工具版本。
6. 将外参写入 URDF/xacro 或静态 TF；不要同时由两个节点发布相同 TF。

矩阵方向必须在使用前检查。若工具输出的是：

```text
T_camera_lidar
```

而 TF 链需要 `T_lidar_camera`，则必须取逆。假设已经知道 `T_base_lidar`，则：

```text
T_base_camera = T_base_lidar * T_lidar_camera
```

不要仅凭输出文件变量名猜方向。使用 LiDAR 点投影到图像验证墙角、桌边、门框等边缘，并检查近、中、远距离和图像四周。静止时对齐、运动时错位通常是时间问题，不应继续手调空间外参。

该标定不会自动给出以下内容：

- Femto Bolt RGB 与 depth 的厂内标定/对齐。
- `base_link -> lidar`。
- Mid-360 LiDAR -> 内置 IMU 外参是否被 LIO 正确使用。
- 相机与 LiDAR 时钟偏移。

### 5.7 时间同步和曝光时刻位姿

相机帧的位姿必须对应曝光时刻，而不是 ROS callback 到达时刻。维护带时间戳的 LIO pose buffer：

```text
translation -> linear interpolation
rotation    -> quaternion SLERP
query time  -> t_camera + calibrated_time_offset
```

ROS `ApproximateTime` 只能配对消息，不能替代时钟同步和位姿插值。建议优先建立共同硬件时钟/PTP/触发方案；做不到时，至少离线估计时间偏移并记录补偿值。

验收方式：

- 静止投影正确，缓慢平移和原地旋转也应正确。
- 旋转时边缘朝同一方向系统拖尾，优先检查时间偏移。
- 超过 `pose_age_ms` 阈值、无法插值或 TF 缺失时，本帧不得进 3DGS，机器人必须停车。

### 5.8 坐标系转换

ROS 常用 `x` 前、`y` 左、`z` 上；当前 IEVE/MinAV/3DGS 沿用 Habitat 的 `y` 上、`x-z` 水平面。为减少对大量 MinAV `position[[0,2]]` 逻辑的修改，第一版可在唯一的 `transforms.py` 中使用：

```text
p_ieve = [-y_ros, z_ros, -x_ros]

R_ieve_from_ros =
[[ 0, -1,  0],
 [ 0,  0,  1],
 [-1,  0,  0]]
```

这样当前 [`get_sim_location()`](../envs/habitat/imagegoal_env.py#L571) 所代表的二维平面与 ROS `[x,y,yaw]` 可以对齐。所有 position、rotation、目标高斯和导航目标必须使用同一个世界变换，不能只交换平移轴。

对 3DGS，建议直接传：

```text
T_ieve_camera_optical = T_ieve_rosworld * T_rosworld_camera_optical
```

然后在 mapper 中明确计算它需要的 `w2c`。ROS optical 已是 `x` 右、`y` 下、`z` 前，不要再无条件套用当前 `_quat_to_matrix()` 中的固定翻转。坐标转换只允许在一个模块发生一次。

必须为以下基准姿态写单元测试：

- 机器人在原点朝 ROS `+x`。
- 向前走 1 m，IEVE 平面 x 增加 1 m。
- 左转 90 度，IEVE yaw 符号与 FMM 转向一致。
- 相机光轴前方 1 m 的点投影到 RGB 中心。
- `T_world_camera * T_camera_world` 为单位阵。

### 5.9 3DGS 和 MinAV 位姿字段

当前 [`update_3dgs_mappers()`](../main.py#L247) 使用：

```text
position = camera_position
rotation = agent_rotation_xyzw
```

对有安装外参的实机这是错误组合。接口应拆为：

```text
T_world_base
T_world_camera_optical
base_position/base_rotation       # FMM、车辆进度、碰撞判断
camera_position/camera_rotation   # 3DGS、视角生成、图像投影
```

同时修改：

- [`main.py::update_3dgs_mappers`](../main.py#L212)：优先传完整相机矩阵。
- [`main.py::build_minav_payload`](../main.py#L980)：同时传 base 与 camera pose，字段不能再混用。
- [`gaussian_mapping/phase1_gaussmi.py`](../gaussian_mapping/phase1_gaussmi.py#L172)：视角评分使用 camera pose，移动进度使用 base pose。
- [`ieve_minav/manager.py`](../ieve_minav/manager.py#L5747)：所有 Gaussian/NBV 调用使用 camera pose，导航距离使用 base position。
- [`gaussian_mapping/mapper/realtime_3dgs.py`](../gaussian_mapping/mapper/realtime_3dgs.py#L213)：`add_frame()` 接受显式 pose matrix。

### 5.10 二维地图和 Mid-360 障碍层

当前有两条基于相机深度的建图链：

- [`Semantic_Mapping`](../model.py#L211) 生成局部 obstacle/explored/semantic map。
- [`update_minav_full_grid_from_depth`](../envs/utils/depth_utils.py#L394) 生成 MinAV full grid。

实机建议保留相机深度用于视觉 explored 和语义投影，同时从 Mid-360 注册点云生成独立导航占据层：

```text
deskewed/registered cloud
 -> 去车体自反射和近距盲区
 -> 地面/高度过滤
 -> 2D occupancy
 -> 按真实 footprint + 安全余量膨胀
 -> nav_occupancy
```

在 [`agents/instance_exp.py::_plan`](../agents/instance_exp.py#L1794) 前，将 `nav_occupancy` 作为 FMM traversibility 的权威障碍层；RGB-D obstacle 可作为补充。不要覆盖 semantic/visual explored。

当前 collision map 主要通过“发过前进命令但位姿没动”推断碰撞，见 [`agents/instance_exp.py`](../agents/instance_exp.py#L1865)。实机可保留它用于 stuck 检测，但不能用它代替 LiDAR 主动避障。

若后续接 Nav2，推荐边界是：IEVE 输出目标点/目标朝向，Nav2 负责全局/局部路径和速度；MinAV 的阶段状态与目标确认仍由 IEVE 管理。第一版为了减少变量，可以先保留 FMM并融合 LiDAR occupancy。

### 5.11 控制和安全

控制回路必须与深度网络/3DGS 推理回路解耦：

```text
IEVE planner: 低频更新目标或 MotionCommand
control node: 20-50 Hz 发布、限速、限加速度
safety node: 高频检查 LiDAR/急停/心跳，并拥有最终停车权
```

第一轮联调建议限制在约 `0.1-0.2 m/s`，角速度根据底盘尺寸和制动能力实测后设置。必须具备：

- 物理急停和软件急停。
- 命令 watchdog；IEVE 进程卡住时自动发零速度。
- RGB-D、LIO、TF、LiDAR occupancy 任一超时即停车。
- 前方制动距离检查和 footprint 碰撞检查。
- 速度/角速度/加速度限制。
- terminal STOP 持续锁存零速度，直到新的 episode 明确开始。
- 启动默认不使能电机；shadow mode 默认不发布运动命令。

当前 `_non_terminal_turn_action()` 通过 Habitat `time_step` 精确转固定角度，见 [`agents/instance_exp.py`](../agents/instance_exp.py#L346)。实机不能假设发一个角速度持续某个理论时长就精确转 30 度；应基于 LIO yaw 闭环，在达到目标角度或超时后停止。

### 5.12 Episode、停止条件和指标

实机 `reset()` 不应重置/瞬移机器人，而应：

1. 锁存零速度。
2. 等待传感器和定位健康。
3. 接收新的目标图。
4. 记录当前 `T_odom_base` 为 episode 原点。
5. 清空 IEVE 地图、3DGS 和 MinAV 状态。
6. 操作者确认后进入 active 状态。

实机 `done` 至少包括：目标导航完成、超时、人工停止、安全故障和定位丢失。Habitat 的 geodesic distance、SPL 和 success 不存在；应单独记录：

- 是否由人工/外部真值确认找到正确实例。
- 最终底盘到目标估计距离。
- 行驶路径长度、耗时、定位中断次数、安全停车次数。
- 算法自己估计的 success 与人工真值 success 分开保存。

[`main.py`](../main.py#L39) 的 Habitat 评测恢复与汇总逻辑在 real backend 下必须禁用。

## 6. 必须按这个先后顺序实施

以下每一步都有验收门槛。上一阶段未通过，不要开启下一阶段的自主运动。

### 阶段 0：冻结仿真基线和安全边界

工作：

- 保存当前可复现的 Habitat 命令、配置、模型和一组 episode 输出。
- 确定机器人 footprint、最大安全速度、制动距离和急停方式。
- 选定 ROS 2 发行版、驱动版本和所有 TF frame 名称。

通过标准：仿真基线可复现；实机断开 IEVE 时也能由独立安全节点停车。

### 阶段 1：只打通传感器驱动和 rosbag

工作：

- 启动 `livox_ros_driver2`，验证点云和 IMU 时间戳/频率。
- 启动 Femto Bolt 驱动，固定 RGB、aligned depth、CameraInfo 模式。
- 同时录制点云、IMU、RGB、depth、CameraInfo、TF 和底盘里程计。

通过标准：连续录制 20-30 分钟无异常跳时、重复帧和断流；深度与 RGB 对齐。

### 阶段 2：先把 Mid-360 定位做稳

工作：

- 配置 FAST-LIO2 的 Mid-360 消息、单点时间、IMU 和内外参。
- 静止初始化，手推/遥控完成直线、原地转动和闭环测试。
- 明确输出 pose 的 child frame，再得到 `T_odom_base`。

通过标准：注册点云无明显双墙/拖影；静止不持续漂移；短程闭环误差满足场地导航要求。

### 阶段 3：完成相机内参和 RGB-D 数据契约

工作：

- 验证 Femto Bolt 内参、畸变、depth scale 和 depth-to-color 对齐。
- 固定运行分辨率；实现 `raw_depth_m + camera_K`。
- 完成深度单位、无效值、resize 后内参单元测试。

通过标准：已知平面距离误差可接受；缩放前后反投影同一点一致；近距离场景不会被误判为 Habitat 归一化深度。

### 阶段 4：做 LiDAR-相机外参和时间同步

工作：

- 使用 `direct_visual_lidar_calibration` 得到 LiDAR-camera 6DoF 外参。
- 补齐 `base_link -> lidar`，形成完整 TF 树。
- 估计时间偏移，并将 LIO pose 插值到相机曝光时刻。

通过标准：静止和缓慢运动时，LiDAR 投影在近/中/远距离均与图像边缘对齐；TF 无环且只有单一发布者。

### 阶段 5：实现 RealRobotEnv，但只做离线回放

工作：

- 增加 backend 分派、实机 observation contract、目标输入和单环境适配器。
- 通过 rosbag 喂给 IEVE，禁止发布 `/cmd_vel`。
- 对同一帧打印时间戳、K、depth range、base pose、camera pose 和数据年龄。

通过标准：长时间回放无 shape/unit/frame 错误；每帧 pose 可插值；丢消息时按设计降级。

### 阶段 6：只验证二维语义地图和 3DGS

工作：

- 修正真实内参、深度和 camera pose 接口。
- 先关闭导航，只构建 IEVE 2D map 和 3DGS。
- 检查地图方向、尺度、墙面厚度、目标 mask 投影和高斯重影。

通过标准：向前移动时地图轨迹方向正确；原地旋转地图不漂移；3DGS 无系统性双墙；目标高斯落在真实物体上。

### 阶段 7：接入 Mid-360 导航障碍层

工作：

- 从注册点云生成 `nav_occupancy`。
- 与视觉 `visual_explored` 分离。
- 校准 footprint、膨胀半径、地面过滤、台阶和悬空障碍阈值。

通过标准：障碍物、窄通道和机器人自身点处理正确；FMM 规划不穿墙；相机未观察区域仍会被视觉 frontier 选择。

### 阶段 8：底盘闭环控制，先不用 IEVE 目标

工作：

- 实现物理单位 MotionCommand、yaw 闭环转向、限速、watchdog 和安全停车。
- 先完成手工给定短目标/固定轨迹。

通过标准：前进/左右转符号正确；命令断开自动停车；急停有效；定位超时不会继续运动。

### 阶段 9：IEVE shadow mode

工作：

- IEVE 实时运行并生成目标、FMM STG 和速度，但不下发。
- 对比人工遥控轨迹，检查规划目标、转向方向和停止判断。

通过标准：连续多次任务不产生越界目标、反向转向、过期命令或穿越障碍的路径。

### 阶段 10：低速闭环，再逐步开启完整算法

开启顺序：

1. 固定点 FMM 导航。
2. frontier 探索。
3. 实例检测和目标图匹配。
4. 3DGS 目标确认。
5. MinAV localization/buffer/navigation 全状态机。
6. 最后才考虑 Nav2 替换局部执行、全局重定位和回环。

每增加一个功能都重新跑传感器中断、定位跳变、障碍突然出现和人工急停测试。

## 7. 最小验收测试矩阵

| 测试 | 期望结果 |
|---|---|
| 深度为 1 m | 2D/3D 点距离均约为 1 m，不做二次缩放 |
| 图像 resize | 投影像素在 resize 前后对应一致 |
| 机器人向前 1 m | `base_pose_2d.x`、IEVE 轨迹和高斯地图同向且尺度一致 |
| 原地左转 90 度 | FMM yaw、相机视角和底盘转向符号一致 |
| 静态场景慢速转动 | RGB-D 高斯无明显重影，LiDAR 投影不拖尾 |
| 拔掉相机 | watchdog 在规定时间内停车 |
| 停止 LIO | 不再消费无有效 pose 的图像，并停车 |
| IEVE 进程卡住 | 独立控制/安全节点自动发零速度 |
| 前方突然放障碍 | LiDAR 安全层早于 IEVE 下一次推理停车 |
| 目标 episode 完成 | terminal stop 锁存，未收到新任务前不恢复运动 |

## 8. 第一版完成的判定

第一版实机迁移完成至少应满足：

- Habitat 和 real backend 可切换，仿真回归未破坏。
- Femto Bolt RGB-D、内参、depth scale 明确且同帧。
- FAST-LIO2 能在相机曝光时刻给出 `T_odom_base`。
- `direct_visual_lidar_calibration` 外参已通过动态投影验证。
- 3DGS 使用完整相机位姿而不是车体旋转。
- IEVE 的导航占据层与视觉探索层分离。
- 所有底盘命令使用物理单位并经过独立安全层。
- rosbag 离线、shadow mode、低速闭环依次通过。
- 实机 success 指标与 Habitat SPL 指标分开记录。

这之后再优化 FAST-LIO2/轮速融合、全局重定位、回环、Nav2 和运行速度。它们不是第一轮把 IEVE 安全搬上实机的前置条件。
