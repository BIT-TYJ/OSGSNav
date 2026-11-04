# IEVE 实机 30 Hz 相机、最新帧缓存与停走式控制实施方案

## 1. 文档目的

本文给出以下实机运行方式的具体改造方案：

```text
RGB-D 相机固定 30 Hz
        |
只保留最新同步帧，旧帧自动丢弃
        |
机器人停车并稳定后，IEVE 取一帧进行计算
        |
IEVE/FMM 输出短期目标 STG
        |
10-20 Hz 底盘闭环执行一小段运动
        |
独立 LiDAR 安全层最终决定是否允许 /cmd_vel 输出
```

目标不是让 IEVE 处理全部 30 Hz 图像，而是保证 IEVE 每次处理的都是最新、同步且位姿正确的图像，同时确保计算时间较长时机器人不会依赖陈旧视觉结果持续运动。

本文按 ROS1/`rospy` 描述，因为当前仓库第三方 GauSS-MI 代码使用 ROS1。若实机使用 ROS2，模块边界和消息字段不变，只需将 subscriber queue、service、actionlib 和时间 API 替换为 ROS2 对应实现。

## 2. 当前代码事实与改造边界

当前 IEVE 是 Habitat 同步循环：

1. `main.py` 处理当前观测并更新二维地图、3DGS 和目标状态。
2. `agents/instance_exp.py::plan_act_and_preprocess()` 计算 FMM 动作。
3. `super().step(action)` 让 Habitat 执行动作并返回下一帧。
4. Habitat 返回的 RGB、Depth、位姿天然属于同一个仿真 step。

实机不能直接复用 Habitat 的动作执行接口，尤其不能把归一化的 `linear_velocity` 和 `angular_velocity` 数组直接发布为 `/cmd_vel`。实机改造应保留以下算法：

- IEVE/MinAV 阶段管理与目标选择。
- 语义分割和实例检测。
- 二维语义地图。
- 3DGS 更新与目标确认。
- FMM 可达性判断和 STG 计算。

实机需要替换或新增以下部分：

- Habitat RGB-D 和真值位姿输入，替换为 ROS RGB-D、CameraInfo 和 Point-LIO 位姿。
- Habitat `super().step(action)`，替换为 STG 发布、实机闭环执行和下一次停车采样。
- Habitat 碰撞反馈，替换为 Mid-360 局部障碍地图和独立安全层。
- 仿真相机内参，替换为真实 `CameraInfo.K`。

## 3. 总体进程与数据流

下面按逻辑模块展示数据流。首版部署时不要真的把大图像在多个 Python 进程之间重复传递；推荐拆成三个进程：

1. `ieve_real_robot` 进程：包含 RGB-D subscriber、最新帧缓存、位姿缓存、Coordinator 和 IEVE Worker。
2. `ieve_stg_controller_node`：独立的 10-20 Hz 底盘闭环进程。
3. `ieve_safety_supervisor_node`：独立安全进程，并且是唯一的最终 `/cmd_vel` publisher。

这样即使 IEVE GPU Worker 卡住，控制和安全进程仍能按超时规则停车。图中的 Buffer 和 Pose Buffer 是 `ieve_real_robot` 内部模块，也可以在离线调试时单独启动成诊断节点。

```text
camera_driver (30 Hz)
  /camera/color/image_rect
  /camera/aligned_depth_to_color/image_raw
  /camera/color/camera_info
                  |
                  v
ieve_rgbd_buffer_node -----------------------+
  RGB/Depth 同步                              |
  最新帧单槽缓存                              |
  不运行神经网络                              |
                  |                          |
                  v                          |
point_lio -> ieve_pose_buffer_node            |
  /point_lio/odom                             |
  位姿历史缓存与曝光时刻插值                   |
                  |                          |
                  +------------+-------------+
                               v
                    ieve_stop_go_coordinator
                    STOPPING/SETTLING/CAPTURE/
                    PROCESSING/EXECUTING 状态机
                               |
                               v
                          IEVE Worker
                    语义、2D map、3DGS、FMM
                               |
                         /ieve/fmm_stg
                               v
                    ieve_stg_controller_node
                     10-20 Hz 闭环控制
                               |
                     /ieve/cmd_vel_raw
                               v
Mid-360 -> ieve_safety_supervisor_node
            障碍、超时、急停、限速
                               |
                           /cmd_vel
                               v
                            底盘驱动
```

关键约束：`ieve_safety_supervisor_node` 必须是正常运行时唯一发布最终 `/cmd_vel` 的节点。IEVE 和 STG Controller 都不能绕过安全层直接控制底盘。

## 4. 建议新增的目录和文件

建议先在当前仓库新增以下结构：

```text
real_robot/
  __init__.py
  interfaces.py                 # 内部不可变数据结构
  latest_rgbd_buffer.py         # 最新同步 RGB-D 单槽缓存
  pose_interpolator.py          # Point-LIO 位姿缓存、线性插值和 SLERP
  coordinate_transform.py       # ROS、IEVE、camera optical 坐标唯一转换点
  real_robot_bridge.py          # ROS 数据转换为 IEVE obs/info
  stop_go_coordinator.py        # 停走状态机
  ieve_worker.py                # 串行调用 IEVE 的工作线程
  fmm_stg_adapter.py            # FMM 栅格 STG 转 odom PoseStamped
  nodes/
    ieve_real_robot_node.py       # Buffer、Pose、Coordinator、Worker 同进程
    rgbd_buffer_diagnostic_node.py # 可选，仅用于独立采集诊断
    stg_controller_node.py
    safety_supervisor_node.py
  config/
    real_robot_stop_go.yaml
  tests/
    test_latest_rgbd_buffer.py
    test_pose_interpolator.py
    test_coordinate_transform.py
    test_stop_go_state_machine.py
    test_stg_grid_to_odom.py

ros/ieve_robot_msgs/
  CMakeLists.txt
  package.xml
  msg/IeveRunState.msg
  msg/StgExecutionStatus.msg
  msg/SafetyStatus.msg
  srv/StartEpisode.srv
  srv/StopEpisode.srv
  srv/ResetSafetyStop.srv
```

`real_robot/` 保存与 IEVE 强相关的 Python 实现；`ros/ieve_robot_msgs/` 只保存跨节点通信需要的自定义消息。图像大数组不通过自定义 service 反复复制，优先在 IEVE 进程内使用单槽缓存。必须跨进程时再使用 ROS nodelet/shared-memory transport。

## 5. 统一参数配置

新增 `real_robot/config/real_robot_stop_go.yaml`，建议第一轮使用以下保守配置：

```yaml
camera:
  expected_fps: 30.0
  sync_slop_sec: 0.010
  max_frame_age_at_capture_sec: 0.100
  settle_time_sec: 0.150
  require_aligned_depth: true
  depth_encoding: "16UC1"
  depth_scale_to_meter: 0.001

pose:
  odom_frame: "odom"
  base_frame: "base_link"
  camera_frame: "camera_color_optical_frame"
  history_sec: 5.0
  max_bracket_gap_sec: 0.050
  max_latest_age_sec: 0.100
  camera_lidar_time_offset_sec: 0.0

stop_go:
  control_rate_hz: 20.0
  stopped_linear_speed_mps: 0.01
  stopped_angular_speed_rps: 0.02
  stopped_confirm_sec: 0.200
  max_motion_duration_sec: 0.500
  max_motion_distance_m: 0.150
  stg_arrival_radius_m: 0.100
  processing_timeout_sec: 120.0

controller:
  rate_hz: 20.0
  align_first_threshold_deg: 20.0
  k_linear: 0.8
  k_angular: 1.2
  max_linear_mps: 0.20
  max_angular_rps: 0.40
  raw_cmd_timeout_sec: 0.200

safety:
  lidar_timeout_sec: 0.200
  odom_timeout_sec: 0.200
  raw_cmd_timeout_sec: 0.200
  hard_stop_margin_m: 0.20
  slow_down_margin_m: 0.50
  max_linear_mps: 0.20
  max_angular_rps: 0.40
  emergency_stop_latched: true
```

这些参数不是所有底盘的通用最终值。机器人 footprint、制动距离、Mid-360 盲区和底盘通信延迟测量完成后，必须重新计算安全距离。

## 6. 模块一：RGB-D 相机固定 30 Hz

### 6.1 改动内容

相机驱动固定在实际标定过的运行模式，例如 640x480、30 Hz。不要因为 IEVE 推理只有 0.5 Hz 就把相机驱动改成 0.5 Hz。

相机驱动至少发布：

| Topic | 消息类型 | 必需字段/约束 |
|---|---|---|
| `/camera/color/image_rect` | `sensor_msgs/Image` | `header.stamp` 为曝光时刻，编码明确为 `rgb8` 或 `bgr8` |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 已对齐到彩色图，推荐 `16UC1` 毫米或 `32FC1` 米 |
| `/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 分辨率、畸变状态和 `K/P` 必须对应当前图像 |

相机参数要求：

- FPS 固定为 30 Hz；若 USB/网络实测无法稳定，可固定为 15 Hz，但仍然不能排队逐帧推理。
- 运行分辨率、裁剪和去畸变方式必须与标定时一致。
- 优先使用 RGB-D 硬件同步和 depth-to-color 对齐。
- 自动曝光可以使用，但应限制最长曝光时间，避免曝光时间接近或超过帧周期导致运动模糊。
- 驱动时间戳必须来自采集/曝光时刻，不能在 ROS 发布回调中用 `now()` 重新打时间戳。

### 6.2 实现效果

- 相机始终提供最新场景，不受 IEVE 推理耗时影响。
- Point-LIO、录像和诊断仍可使用相机原始 30 Hz 数据。
- IEVE 可以主动选择停车稳定后的新帧，而不是处理积压旧帧。

### 6.3 验收标准

- 连续 60 秒实际频率稳定在设备允许误差内，例如 30 Hz 模式下约 29-31 Hz。
- `header.stamp` 单调递增，无大幅跳变。
- RGB 与对齐深度尺寸完全一致。
- 彩色图像任一点 `(u, v)` 与深度图 `(u, v)` 表示同一视线。

## 7. 模块二：RGB-D 同步与最新帧单槽缓存

### 7.1 改动内容

新增 `real_robot/latest_rgbd_buffer.py`。该模块只负责同步、校验和保存最新帧，禁止执行 Detectron2、3DGS 或 IEVE。

ROS1 subscriber 建议：

```python
rgb_sub = message_filters.Subscriber(
    "/camera/color/image_rect", Image, queue_size=1, buff_size=2**24
)
depth_sub = message_filters.Subscriber(
    "/camera/aligned_depth_to_color/image_raw", Image,
    queue_size=1, buff_size=2**24
)
sync = message_filters.ApproximateTimeSynchronizer(
    [rgb_sub, depth_sub], queue_size=5, slop=0.010,
    allow_headerless=False,
)
```

硬件已经给 RGB 和 Depth 相同时间戳时，改用 `TimeSynchronizer`。`queue_size=5` 只是同步器短暂配对窗口；配对完成后的应用缓存仍然只能保存一帧。

`CameraInfo` 通常在相机模式不变时保持固定，可以使用独立 `queue_size=1` subscriber 保存最新值。每个 RGB-D pair 进入缓存前，必须检查 CameraInfo 的 `width/height` 与图像一致；相机模式或内参发生变化时应拒绝继续运行并要求重新初始化 3DGS，不能在同一个高斯地图中静默切换内参。

内部数据结构定义在 `real_robot/interfaces.py`：

```python
@dataclass(frozen=True)
class RgbdFrame:
    seq: int
    rgb_stamp_sec: float
    depth_stamp_sec: float
    capture_stamp_sec: float
    rgb: np.ndarray             # uint8, H x W x 3, RGB
    depth_m: np.ndarray         # float32, H x W, meter, invalid=0
    camera_info: CameraIntrinsics
    rgb_frame_id: str
    depth_frame_id: str
```

缓存接口：

```python
class LatestRgbdBuffer:
    def push(self, frame: RgbdFrame) -> None: ...

    def take_latest(
        self,
        after_seq: int,
        not_before_stamp_sec: float,
        timeout_sec: float,
    ) -> RgbdFrame: ...

    def stats(self) -> dict: ...
```

行为约束：

- `push()` 使用锁原子替换 `_latest`，旧对象立即释放，不使用无界 `Queue`。
- `take_latest()` 只返回 `seq > after_seq` 且采集时间晚于 `not_before_stamp_sec` 的帧。
- Worker 取得帧后获得不可变快照；新回调可以继续覆盖缓存，而不能修改 Worker 正在处理的数组。
- 统计 `received_pairs`、`overwritten_pairs`、`rejected_pairs`、`last_seq`、`last_stamp` 和 RGB-Depth 时间差。
- 深度在进入缓存时统一成米，`NaN`、`Inf`、负值和超量程值统一为 0。

不要把 Point-LIO Odometry 放进 RGB/Depth `message_filters` 一起做三路近似同步。位姿需要在相机曝光时刻插值，最近邻配对会在转弯时产生系统误差。

### 7.2 实现效果

假设相机 30 Hz、IEVE 单帧耗时 2 秒：

- 2 秒内相机产生约 60 帧。
- 缓存只保留第 60 帧，不积压前 59 帧。
- IEVE 下一次开始时处理最新帧，数据延迟接近一个相机周期，而不是累计 2 秒、4 秒或更长。

### 7.3 验收标准

- IEVE 故意暂停 10 秒后，缓存深度仍为 1。
- 恢复 IEVE 时取得的是暂停结束附近的新帧，不是暂停前的旧帧。
- 正常运行时 `capture_start_wall_time - frame.capture_stamp` 小于配置的 100 ms。
- RGB 与 Depth 时间差不超过 10 ms；超限帧不得进入 IEVE。

## 8. 模块三：Point-LIO 位姿缓存与曝光时刻插值

### 8.1 输入接口

| Topic/TF | 消息类型 | 用途 |
|---|---|---|
| `/point_lio/odom` | `nav_msgs/Odometry` | 连续保存 `T_odom_lidar(t)` 或 `T_odom_imu(t)` |
| `/tf_static` | `tf2_msgs/TFMessage` | `base_link`、LiDAR、camera link/optical 外参 |
| `/tf` | `tf2_msgs/TFMessage` | 如 Point-LIO 通过 TF 发布动态位姿则读取该接口 |

必须先确认 `/point_lio/odom.child_frame_id` 代表 IMU、LiDAR 还是 `base_link`，不能仅根据 topic 名假定坐标系。

### 8.2 改动内容

新增 `real_robot/pose_interpolator.py`：

```python
@dataclass(frozen=True)
class TimedPose:
    stamp_sec: float
    translation_xyz: np.ndarray       # meter
    rotation_xyzw: np.ndarray         # normalized quaternion

class PoseInterpolator:
    def push_odom(self, pose: TimedPose) -> None: ...

    def lookup(self, stamp_sec: float) -> TimedPose: ...
```

实现要求：

- 保存最近 5 秒或足够覆盖 IEVE 取帧延迟的有序位姿。
- 对目标时刻两侧位姿做 bracket lookup。
- 平移采用线性插值。
- 旋转采用 quaternion SLERP，并统一四元数符号避免长路径插值。
- 目标时刻需加上标定得到的 `camera_lidar_time_offset_sec`。
- 两侧位姿间隔超过 `max_bracket_gap_sec` 时拒绝该帧。
- 禁止长期外推；只有明确测量并限制误差后，才允许几毫秒级短外推。

相机位姿计算：

```text
T_odom_camera(t_c)
  = T_odom_lidar(t_c) * T_lidar_camera
```

或者当 Point-LIO 输出 IMU 位姿时：

```text
T_odom_camera(t_c)
  = T_odom_imu(t_c) * T_imu_lidar * T_lidar_camera
```

所有 ROS、camera optical、IEVE 和 3DGS 轴变换只能在 `coordinate_transform.py` 中实现一次，并为已知单位向量和已知 yaw 编写单元测试。

### 8.3 输出给 IEVE 的接口

每个处理帧形成不可变的 `IeveFrameBundle`：

```python
@dataclass(frozen=True)
class IeveFrameBundle:
    frame: RgbdFrame
    base_pose_at_capture: TimedPose
    camera_pose_at_capture: TimedPose
    T_odom_base: np.ndarray            # 4 x 4
    T_odom_camera_optical: np.ndarray  # 4 x 4
```

### 8.4 实现效果

- 图像、深度、mask 和相机位姿严格同帧。
- 推理即使耗时数秒，也不会错误地把“推理完成时位姿”赋给历史图像。
- 车辆转弯时目标投影偏移和 3DGS 双边重影显著减少。

## 9. 模块四：停走状态机

### 9.1 状态定义

新增 `real_robot/stop_go_coordinator.py`，所有移动和采集都由以下状态机协调：

```text
IDLE
  | StartEpisode
  v
STOPPING <-----------------------------------------+
  | 速度连续 stopped_confirm_sec 低于阈值          |
  v                                                 |
SETTLING                                             |
  | 等待相机/机身振动稳定 settle_time_sec            |
  v                                                 |
CAPTURE                                              |
  | 取得一帧 stamp >= settling 完成时刻的新 RGB-D    |
  v                                                 |
PROCESSING                                           |
  | IEVE/3DGS/FMM 成功                               |
  v                                                 |
PLAN_READY                                           |
  | 发布 STG                                         |
  v                                                 |
EXECUTING -- 到达/0.5 s/0.15 m/安全停止/错误 --------+
```

任何状态收到人工急停、传感器超时、节点异常或 `StopEpisode` 都进入 `FAULT_STOP`，持续要求零速度。

### 9.2 状态消息接口

`ros/ieve_robot_msgs/msg/IeveRunState.msg`：

```text
std_msgs/Header header
uint8 IDLE=0
uint8 STOPPING=1
uint8 SETTLING=2
uint8 CAPTURE=3
uint8 PROCESSING=4
uint8 PLAN_READY=5
uint8 EXECUTING=6
uint8 FAULT_STOP=7
uint8 state
uint32 cycle_id
uint64 selected_frame_seq
time selected_frame_stamp
float32 inference_elapsed_sec
string detail
```

Coordinator 发布：

| Topic | 类型 | 说明 |
|---|---|---|
| `/ieve/run_state` | `ieve_robot_msgs/IeveRunState` | 10 Hz 发布当前状态和周期 |
| `/ieve/motion_allowed` | `std_msgs/Bool` | 只有 EXECUTING 为 true；安全层仍可否决 |
| `/ieve/processing_heartbeat` | `std_msgs/Header` | PROCESSING 中持续发布进程存活信息 |

Coordinator 订阅：

| Topic | 类型 | 说明 |
|---|---|---|
| `/ieve/stg_execution_status` | `ieve_robot_msgs/StgExecutionStatus` | STG 执行进度和结束原因 |
| `/ieve/safety_status` | `ieve_robot_msgs/SafetyStatus` | 安全层状态 |
| `/point_lio/odom` | `nav_msgs/Odometry` | 判断车辆是否真正停止 |

Episode service：

```text
# StartEpisode.srv
string episode_name
string target_image_path
---
bool accepted
string message

# StopEpisode.srv
string reason
---
bool stopped
string message

# ResetSafetyStop.srv
string operator
string reason
---
bool reset
string message
```

### 9.3 停车确认规则

进入 `STOPPING` 后不能只发布一次零速度就开始拍照。必须同时满足：

```text
abs(base_linear_speed)  < 0.01 m/s
abs(base_angular_speed) < 0.02 rad/s
连续保持时间           >= 0.20 s
LiDAR 安全状态          == OK
```

然后进入 `SETTLING` 再等待约 0.15 秒。`CAPTURE` 只能取 `capture_stamp >= settle_complete_stamp` 的帧，防止误用缓存中机器人运动期间的图像。

### 9.4 实现效果

- IEVE 计算期间底盘保持停止。
- 每个 IEVE 输入帧都来自车辆稳定后的新采样。
- 单次动作距离和时间受限，即使高层计划错误也不会长时间开环运动。
- 初始 360 度观察应拆成多次小转动，每次转动后停车、稳定、采集，而不是持续旋转期间进行慢推理。

## 10. 模块五：IEVE Worker 与真实观测适配

### 10.1 改动入口

新增命令行参数：

```text
--runtime_backend habitat|real
--real_robot_config real_robot/config/real_robot_stop_go.yaml
```

修改 `envs/__init__.py::make_vec_envs(args)`：

```python
if args.runtime_backend == "real":
    return make_real_robot_env(args)
return make_habitat_vec_envs(args)
```

实机首版强制：

```text
num_processes = 1
eval = 1
```

实机模式不能创建 Habitat Simulator，也不能读取 Habitat 真值位姿或 Habitat semantic sensor。

### 10.2 RealRobotEnv 接口

新增 `RealRobotEnv`，对主循环提供与当前环境相近的接口：

```python
class RealRobotEnv:
    def reset(self) -> tuple[np.ndarray, dict]: ...

    def plan_act_and_preprocess(
        self, planner_inputs: list[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]: ...

    def close(self) -> None: ...
```

`reset()` 行为：

1. 请求 Coordinator 停车。
2. 等待 STOPPED 和 SETTLING 完成。
3. 取得最新 `IeveFrameBundle`。
4. 调用现有实例检测/预处理逻辑。
5. 返回第一帧 `obs` 和 `info`。

`plan_act_and_preprocess()` 行为：

1. 使用当前 `planner_inputs` 运行 FMM，得到 STG、stop 和 replan。
2. 不调用 Habitat `super().step(action)`。
3. 将 STG 转换成 `/ieve/fmm_stg`。
4. 等待 Coordinator 完成一次受限运动并重新停车采样。
5. 返回新的真实 `obs/info`。

### 10.3 IEVE `info` 字段契约

RealRobot Bridge 必须生成：

| 字段 | 类型/单位 | 来源 |
|---|---|---|
| `raw_rgb` | `uint8 HxWx3`, RGB | 校正后彩色图 |
| `raw_depth` | `float32 HxW`, meter | 对齐深度，无效为 0 |
| `ieve_gt_pose` | `[x_odom, y_odom, yaw_rad]` | 图像曝光时刻的 `T_odom_base`；字段名后续可重构 |
| `agent_position` | `[x,y,z]`, meter | 图像曝光时刻车体位置，转换到 IEVE 世界轴 |
| `agent_rotation_xyzw` | quaternion | 图像曝光时刻车体姿态，IEVE 世界轴 |
| `camera_position` | `[x,y,z]`, meter | 图像曝光时刻相机光心，IEVE 世界轴 |
| `camera_rotation_xyzw` | quaternion | 相机完整姿态，包含安装外参 |
| `T_odom_camera_optical` | `float64 4x4` | 3DGS 的显式相机位姿源 |
| `sensor_stamp_sec` | float64 | 相机曝光中点时间 |
| `pose_stamp_sec` | float64 | 实际插值查询时刻 |
| `frame_seq` | uint64 | 最新帧缓存序号 |
| `camera_intrinsics` | `{fx,fy,cx,cy,w,h}` | 当前 `CameraInfo` |
| `sensor_pose` | `[dx,dy,dyaw]` | 与上一处理帧的相对运动 |
| `agent_height` | meter | 相机相对任务起始地面的高度 |

`obs` 继续使用当前主循环预期的 `C x H x W` float tensor。RGB 和深度如何降采样可以复用 `_preprocess_obs()`，但原始 RGB-D 必须保留在 `info` 中供 3DGS 使用。

### 10.4 需要修改的 3DGS 接口

当前 `main.py::build_3dgs_camera_params()` 使用 `hfov` 推导内参。实机模式修改为读取 `info["camera_intrinsics"]`：

```python
if args.runtime_backend == "real":
    camera_params = infos[0]["camera_intrinsics"]
else:
    camera_params = build_3dgs_camera_params(args)
```

当前 `update_3dgs_mappers()` 使用 `agent_rotation_xyzw`。修改优先级：

```python
rotation = info.get(
    "camera_rotation_xyzw",
    info.get("agent_rotation_xyzw"),
)
```

更稳妥的最终接口是让 3DGS 直接接收 `T_odom_camera_optical`，避免在 mapper 内再次猜测相机安装旋转。

### 10.5 Worker 调度规则

IEVE Worker 必须是单消费者：

```python
while running:
    request = coordinator.wait_for_capture_request()
    frame = rgbd_buffer.take_latest(
        after_seq=last_processed_seq,
        not_before_stamp_sec=request.settle_complete_stamp,
        timeout_sec=1.0,
    )
    bundle = bridge.make_bundle(frame)
    result = ieve.process(bundle)
    coordinator.submit_plan(result)
    last_processed_seq = frame.seq
```

不得启动多个 Worker 并发修改同一个 2D map、3DGS 或 MinAV manager。相机 callback 与 Worker 可以并行，但算法状态更新必须串行。

### 10.6 实现效果

- Habitat 和实机共享高层 IEVE/FMM 算法，传感器和执行后端明确分离。
- 推理耗时不会在 ROS callback 中阻塞相机采集。
- 真实内参、真实 RGB-D、曝光时刻真实相机位姿进入 3DGS。

## 11. 模块六：FMM STG 到 odom 目标接口

### 11.1 FMM 输出语义化

建议将 `agents/instance_exp.py::_plan()` 拆成两层：

```python
def compute_fmm_result(planner_inputs) -> FmmResult:
    # 共享：返回 stg、stop、replan 和调试信息

def fmm_result_to_habitat_action(result) -> dict:
    # 仅仿真：返回归一化 velocity_control action

def fmm_result_to_odom_goal(result, map_context) -> PoseStamped:
    # 仅实机：返回物理 odom STG
```

内部结构：

```python
@dataclass(frozen=True)
class FmmResult:
    local_stg_row: int
    local_stg_col: int
    stop: bool
    replan: bool
    goal_source: str
    target_mode: str
    map_step: int
```

这样可以防止实机误用 Habitat 归一化速度。

### 11.2 栅格到 odom 的转换

当前二维地图中 row 对应平面 `y`，column 对应平面 `x`。设：

```text
global_row = local_stg_row + gx1
global_col = local_stg_col + gy1
resolution_m = map_resolution_cm / 100

x_internal_map = (global_col + 0.5) * resolution_m
y_internal_map = (global_row + 0.5) * resolution_m
```

当前 `ieve_full_pose_from_gt()` 将 episode 起点放在地图中心，因此转换回 `odom`：

```text
x_odom = episode_origin_x + x_internal_map - map_center_m
y_odom = episode_origin_y + y_internal_map - map_center_m
```

该公式必须通过以下单元测试后才能上车：

- 机器人在 episode 起点时对应地图中心。
- 地图向右一格使 `x_odom` 增加一个 resolution。
- 地图向下一格使 `y_odom` 增加一个 resolution。
- 给定正前方 STG 时控制器生成正线速度，而不是倒车或横向运动。

### 11.3 ROS 输出

| Topic | 类型 | 内容 |
|---|---|---|
| `/ieve/fmm_stg` | `geometry_msgs/PoseStamped` | `frame_id=odom`，位置是物理米，orientation 表示从当前车体指向 STG 的 yaw |
| `/ieve/fmm_stop` | `std_msgs/Bool` | FMM 是否到达当前最终目标 |
| `/ieve/fmm_replan` | `std_msgs/Bool` | 当前路径是否要求重规划 |

`PoseStamped.header.stamp` 是计划生成时间，不是输入图像采集时间。输入图像时间仍保存在 IEVE 状态和日志中。

### 11.4 实现效果

- FMM 继续决定“往哪里走”。
- 实机控制器只接收物理 odom 目标，不理解 IEVE 栅格内部细节。
- Habitat 与实机速度单位不会混用。

## 12. 模块七：10-20 Hz STG 底盘闭环控制

### 12.1 输入输出接口

`ieve_stg_controller_node`：

订阅：

| Topic | 类型 | 说明 |
|---|---|---|
| `/ieve/fmm_stg` | `geometry_msgs/PoseStamped` | odom 下当前 STG |
| `/ieve/fmm_stop` | `std_msgs/Bool` | true 时立即请求停止 |
| `/ieve/motion_allowed` | `std_msgs/Bool` | 非 EXECUTING 状态必须为 false |
| `/point_lio/odom` 或 TF | `nav_msgs/Odometry` | 当前 `T_odom_base(now)` |

发布：

| Topic | 类型 | 说明 |
|---|---|---|
| `/ieve/cmd_vel_raw` | `geometry_msgs/TwistStamped` | 未经过安全层的期望速度 |
| `/ieve/stg_execution_status` | `ieve_robot_msgs/StgExecutionStatus` | 执行进度 |

`StgExecutionStatus.msg`：

```text
std_msgs/Header header
uint8 IDLE=0
uint8 ALIGNING=1
uint8 ADVANCING=2
uint8 ARRIVED=3
uint8 LIMIT_REACHED=4
uint8 BLOCKED=5
uint8 ABORTED=6
uint8 status
float32 distance_to_stg_m
float32 yaw_error_rad
float32 motion_elapsed_sec
float32 motion_distance_m
string reason
```

### 12.2 控制律

控制频率建议 20 Hz，首轮最大速度限制为 `0.2 m/s` 和 `0.4 rad/s`：

```text
dx = stg_x - base_x
dy = stg_y - base_y
distance = hypot(dx, dy)
target_yaw = atan2(dy, dx)
yaw_error = normalize(target_yaw - base_yaw)

if not motion_allowed or fmm_stop:
    v = 0
    omega = 0
elif abs(yaw_error) > 20 deg:
    v = 0
    omega = clip(k_angular * yaw_error, -omega_max, omega_max)
else:
    v = clip(k_linear * distance, 0, v_max) * max(0, cos(yaw_error))
    omega = clip(k_angular * yaw_error, -omega_max, omega_max)
```

任一条件满足即结束本次 EXECUTING：

- `distance_to_stg <= 0.10 m`。
- 本次运动持续时间达到 `0.50 s`。
- 本次累计运动距离达到 `0.15 m`。
- FMM stop、Coordinator 取消、LiDAR 安全停止或里程计超时。

达到单次运动上限不表示任务失败，只表示本次停走周期结束。Coordinator 随后停车、采集新帧并重新规划。

### 12.3 实现效果

- IEVE 更新较慢时，控制器仍以 10-20 Hz 使用实时 odom 闭环跟踪短期目标。
- 单次动作不会像 Habitat 那样依赖固定归一化速度含义。
- 每次只移动很短距离，视觉结果陈旧的空间范围受到限制。

## 13. 模块八：独立 Mid-360 LiDAR 安全层

### 13.1 职责边界

安全层不运行 IEVE、不等待 3DGS，也不依赖目标检测结果。即使 IEVE 卡死，安全层仍必须能够停止机器人。

安全层输入：

| Topic | 类型 | 说明 |
|---|---|---|
| `/ieve/cmd_vel_raw` | `geometry_msgs/TwistStamped` | STG Controller 原始命令 |
| `/livox/lidar` 或注册点云 | `livox_ros_driver2/CustomMsg` 或 `sensor_msgs/PointCloud2` | 近障判断 |
| `/local_costmap` | `nav_msgs/OccupancyGrid` | 可选，膨胀后的局部障碍地图 |
| `/point_lio/odom` | `nav_msgs/Odometry` | 里程计健康与速度 |
| `/ieve/motion_allowed` | `std_msgs/Bool` | 状态机运动许可 |
| `/emergency_stop` | `std_msgs/Bool` | 硬件或人工急停 |

安全层输出：

| Topic | 类型 | 说明 |
|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | 唯一最终底盘速度输出 |
| `/ieve/safety_status` | `ieve_robot_msgs/SafetyStatus` | 安全状态和原因 |

`SafetyStatus.msg`：

```text
std_msgs/Header header
uint8 OK=0
uint8 SLOW_DOWN=1
uint8 OBSTACLE_STOP=2
uint8 SENSOR_TIMEOUT=3
uint8 COMMAND_TIMEOUT=4
uint8 EMERGENCY_STOP=5
uint8 INTERNAL_ERROR=6
uint8 status
bool motion_permitted
bool latched
float32 nearest_obstacle_m
float32 applied_linear_scale
string reason
```

### 13.2 安全逻辑

每个控制周期按以下优先级执行：

```text
if emergency_stop or latched_fault:
    output zero
elif LiDAR/odom/raw_cmd timeout:
    output zero
elif not motion_allowed:
    output zero
elif obstacle enters hard-stop swept footprint:
    output zero
elif obstacle enters slowdown region:
    scale linear velocity toward zero
else:
    clamp velocity and forward raw command
```

障碍区域不能只用一个固定圆形距离。应根据：

- 机器人真实 footprint。
- 当前前进或转向方向。
- 最大速度和实测制动距离。
- LiDAR 安装位置和近距离盲区。
- 额外 `hard_stop_margin_m`。

构造 swept footprint。人工急停和内部严重错误建议锁存，必须调用 `/ieve/reset_safety_stop` 且物理急停已经释放后才能恢复。普通临时障碍可以在障碍离开并持续稳定一段时间后自动解除，但 Coordinator 仍应先重新进入 STOPPING。

### 13.3 发布者所有权

启动时检查：

```bash
rostopic info /cmd_vel
```

正常模式只能看到安全层一个 publisher。Nav2、键盘遥控、IEVE Controller 和厂商 demo 不能同时发布 `/cmd_vel`。调试遥控应通过明确的 mux，并让急停拥有最高优先级。

### 13.4 实现效果

- IEVE 推理崩溃、GPU 卡住或 ROS 通信超时时机器人自动停车。
- 视觉漏检障碍物时，Mid-360 仍提供独立的最后防线。
- 所有限速和停车原因可诊断、可记录。

## 14. 模块间完整时序

一次正常停走周期如下：

```text
1. Coordinator -> motion_allowed=false
2. Safety Supervisor -> /cmd_vel=0
3. Coordinator 检查 odom 速度连续 0.2 s 为零
4. Coordinator 等待额外 0.15 s 稳定时间
5. Coordinator 记录 settle_complete_stamp
6. LatestRgbdBuffer 返回第一帧：
     seq > last_processed_seq
     capture_stamp >= settle_complete_stamp
     frame_age < 0.1 s
7. PoseInterpolator 查询 T_odom_base(capture_stamp)
8. Bridge 计算 T_odom_camera(capture_stamp)，形成 IeveFrameBundle
9. IEVE Worker 串行执行：
     detector -> semantic map -> 3DGS -> MinAV -> FMM
10. FMM Adapter 发布 /ieve/fmm_stg
11. Coordinator -> state=EXECUTING, motion_allowed=true
12. STG Controller 以 20 Hz 发布 /ieve/cmd_vel_raw
13. Safety Supervisor 每周期检查 LiDAR/odom/timeout 后发布 /cmd_vel
14. 达到 0.5 s、0.15 m、STG 或安全条件后结束
15. Coordinator -> motion_allowed=false，回到 STOPPING
```

在第 9 步 IEVE 处理期间，相机仍以 30 Hz 运行并持续覆盖单槽缓存，但这些帧不会进入队列。第 15 步以后必须重新等待停车稳定，再选一帧新的图像。

## 15. 超时和异常处理矩阵

| 故障 | 检测位置 | 立即动作 | 恢复条件 |
|---|---|---|---|
| RGB-D 超过 100 ms 没有新帧 | Buffer/Coordinator | 保持停车，不进入 PROCESSING | 收到时间戳正常的新同步帧 |
| RGB-Depth 时间差超过 10 ms | Buffer | 丢弃该帧 | 下一组同步帧正常 |
| 找不到曝光时刻两侧 LIO 位姿 | PoseInterpolator | 丢弃该帧并保持停车 | 位姿历史重新覆盖相机时刻 |
| IEVE 计算异常 | Worker | FAULT_STOP，记录 traceback | 人工确认后重启 episode |
| IEVE 超过 processing timeout | Coordinator | FAULT_STOP | 重启 Worker/进程 |
| STG 超时或无效 | Controller | raw cmd 置零，ABORTED | IEVE 重新规划 |
| `/cmd_vel_raw` 超时 | Safety Supervisor | 最终 `/cmd_vel=0` | 新命令和状态机重新授权 |
| Point-LIO 超时 | Safety Supervisor | 最终 `/cmd_vel=0` | LIO 恢复且 Coordinator 重走 STOPPING |
| Mid-360 超时 | Safety Supervisor | 最终 `/cmd_vel=0` | 点云恢复并重新确认安全 |
| 前方障碍进入硬停止区 | Safety Supervisor | 最终 `/cmd_vel=0` | 障碍清除；必要时重新规划 |
| 人工急停 | Safety Supervisor | 锁存零速度 | 物理释放并调用 reset service |

## 16. 日志与性能指标

每个周期至少记录以下字段，建议写入 rosbag 和结构化 CSV/JSON：

```text
cycle_id
run_state
rgb_seq
rgb_stamp
depth_stamp
rgb_depth_delta_ms
capture_frame_age_ms
pose_before_stamp
pose_after_stamp
pose_bracket_gap_ms
selected_T_odom_base
selected_T_odom_camera
detector_time_ms
semantic_map_time_ms
3dgs_time_ms
minav_time_ms
fmm_time_ms
total_processing_time_ms
stg_x_odom
stg_y_odom
motion_time_ms
motion_distance_m
raw_cmd_v
raw_cmd_omega
final_cmd_v
final_cmd_omega
safety_status
cycle_end_reason
buffer_received_count
buffer_overwritten_count
```

重点监控的不是“IEVE 是否达到 30 FPS”，而是：

- IEVE 开始处理时图像是否足够新。
- RGB、Depth 和 pose 是否同一时刻。
- PROCESSING 期间底盘是否确实为零速度。
- 单次运动是否没有超过时间和距离限制。
- 任何数据超时时最终 `/cmd_vel` 是否及时变为零。

## 17. 分阶段实施与验收

### 阶段 A：只接传感器，不接电机

1. 相机固定 30 Hz，验证 RGB-D 和 CameraInfo。
2. 实现 LatestRgbdBuffer，故意让 Worker sleep 5 秒验证无积压。
3. 接入 Point-LIO 位姿缓存和插值。
4. 录制 rosbag，离线检查 RGB-D-pose bundle。

验收：帧缓存恒为 1；选帧年龄小于 100 ms；RGB-Depth 时间差小于 10 ms；位姿插值有完整 bracket。

### 阶段 B：只运行 IEVE，不让车运动

1. 固定底盘或抬起驱动轮。
2. RealRobotEnv 输入真实 RGB-D 和 LIO pose。
3. 验证二维地图方向、3DGS 相机方向和 FMM STG。
4. 检查 `camera_rotation_xyzw` 和 CameraInfo 是否真正被使用。

验收：地图无明显旋转/镜像；静止目标没有持续漂移；3DGS 无明显双墙；STG odom 坐标方向正确。

### 阶段 C：只测试 Controller 和 Safety

1. 不启动 IEVE，人工发布近距离 STG。
2. 最大线速度先限制为 `0.05 m/s`。
3. 测试转向符号、到达半径和单次 0.5 秒限制。
4. 拔掉 LiDAR、停止 odom、停止 raw cmd publisher，逐项验证自动停车。
5. 在软障碍物和人工保护下测量实际制动距离。

验收：所有超时均在规定时间内输出零速度；`/cmd_vel` 只有安全层发布；急停可靠锁存。

### 阶段 D：完整停走循环

1. 限速 `0.05 m/s`，单次移动不超过 5 cm。
2. 验证 STOPPING -> SETTLING -> CAPTURE -> PROCESSING -> EXECUTING。
3. 确认 PROCESSING 全程底盘速度为零。
4. 逐步提高到 `0.1 m/s`，最终不超过首版建议的 `0.2 m/s`。

验收：连续完成至少 50 个停走周期，无图像积压、无过期位姿、无越过单次运动限制、无未解释的安全停车。

## 18. 后续切换连续运动的条件

只有满足以下条件后再考虑连续运动：

- 关键 IEVE 更新稳定在几百毫秒级，而不是偶发数秒或数十秒。
- RGB-D、LIO 和相机外参的动态投影已经通过验证。
- 低层局部控制和 LiDAR 避障能够在不依赖 IEVE 的情况下稳定运行。
- 高层结果带采集时间戳，并能将迟到结果按历史 pose 正确写入地图。
- 已实现 plan age、map age 和 target age 的明确有效期。

连续模式也不应要求 IEVE 处理 30 Hz 全部图像。推荐保持：

```text
相机采集：30 Hz
LIO/安全/底盘闭环：高频持续运行
IEVE/3DGS：最新关键帧、低频异步更新
FMM：地图或目标变化时重规划
```

停走模式和连续模式应通过配置切换，不能删除安全层或改成多个节点直接竞争 `/cmd_vel`。

## 19. 最小实现顺序总结

按依赖关系，建议实际编码顺序为：

1. `LatestRgbdBuffer`：30 Hz 输入、单槽覆盖、同步和统计。
2. `PoseInterpolator`：曝光时刻位姿查询及单元测试。
3. `RealRobotBridge`：生成完整 IEVE `obs/info` 契约。
4. 3DGS 真实 CameraInfo 和 `camera_rotation_xyzw` 接口。
5. FMM `_plan()` 拆分，输出物理 `PoseStamped` STG。
6. `StgController`：20 Hz、物理单位、单次时间/距离限制。
7. `SafetySupervisor`：唯一 `/cmd_vel` publisher 和所有超时。
8. `StopGoCoordinator`：串联停车、稳定、采集、计算和短距离执行。
9. 依次完成离线、静止、低速和完整系统验收。

完成以上改造后的最终功能是：相机保持稳定 30 Hz，IEVE 永远处理最新且同步的停车帧；计算再慢也不会造成图像积压；机器人只在明确授权下短距离闭环移动；IEVE、相机、Point-LIO 或通信任一模块异常时，独立 LiDAR 安全层都能让底盘停止。
