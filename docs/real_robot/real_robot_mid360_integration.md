# IEVE 高斯实例图像导航实机部署与 Livox Mid-360 融合方案

## 1. FMM 问题：保留现有 FMM 并控制实机 `/cmd_vel`

### 1.1 结论与职责边界

实机第一版可以继续使用当前 IEVE 的 FMM，不要求改成 Nav2。正确的控制链路是：

```text
IEVE/MinAV 选择目标
        |
FMM 根据目标栅格、障碍栅格和 T_odom_base 计算短期目标 STG
        |
实机闭环控制器根据 STG 和实时底盘位姿计算 v、omega
        |
发布 geometry_msgs/Twist 到 /cmd_vel
        |
底盘驱动将 /cmd_vel 转换成轮速和电机控制
```

各模块职责如下：

- IEVE/MinAV 决定探索或导航目标。
- FMM 负责在二维可通行栅格中计算距离场、路径方向和短期目标 STG，解决“往哪里走”。
- 实机闭环控制器把 STG 转换为物理单位的线速度和角速度，解决“怎样控制底盘走过去”。
- 底盘驱动负责把 `/cmd_vel` 转换成轮速或电机指令。
- 独立安全层负责限速、数据超时、近距离障碍和紧急停车。

Nav2 不是必需条件。如果后续选择 Nav2，应让 IEVE 只输出目标点并由 Nav2 独占 `/cmd_vel`；不能让现有 FMM 控制器和 Nav2 同时发布底盘速度。

### 1.2 当前 FMM 已经完成的工作

当前 `agents/instance_exp.py:1987` 调用 `_get_stg()`，输入障碍地图、当前栅格和目标栅格，得到：

```text
stg：短期目标栅格 (row, col)
stop：是否已经到达 FMM 最终目标
replan：当前路径是否需要重新规划
```

`agents/instance_exp.py:2089-2167` 随后根据 STG 的方向生成前进、转向和停止动作。因此当前算法已经包含 FMM 规划和一个确定性局部策略，不需要为了“能够规划路径”而额外增加 Nav2。

但这些动作是 Habitat 专用的归一化控制量。例如部分转向逻辑使用 `linear_velocity=-1` 表示仿真控制器中的零线速度，而不是实机的 `-1 m/s`。`agents/instance_exp.py:1182` 最终调用 `super().step(action)` 让 Habitat 执行动作。实机模式不能直接复用这两个接口，也不能把数组原样发布成 `Twist`。

### 1.3 从 FMM STG 生成 `/cmd_vel`

实机控制器应接收 `odom` 坐标系下的 STG。当前 FMM 中 `row` 对应地图 `y`，`col` 对应地图 `x`；应在 ROS Bridge 中使用明确的地图原点、分辨率和坐标变换，将 STG 转换为：

```text
target = [target_x_odom, target_y_odom]
```

控制器持续查询 `T_odom_base`，得到底盘的 `base_x`、`base_y` 和 `base_yaw`，然后计算：

```text
dx = target_x_odom - base_x
dy = target_y_odom - base_y
distance = sqrt(dx^2 + dy^2)
target_yaw = atan2(dy, dx)
yaw_error = normalize(target_yaw - base_yaw)
```

首版可以使用“先对准、再前进”的闭环控制：

```text
如果 FMM stop、传感器超时或安全层触发：
    v = 0
    omega = 0

如果 abs(yaw_error) > 20 deg：
    v = 0
    omega = clip(k_omega * yaw_error, -omega_max, omega_max)

否则：
    v = clip(k_v * distance, 0, v_max) * max(0, cos(yaw_error))
    omega = clip(k_omega * yaw_error, -omega_max, omega_max)
```

建议首轮调试使用保守参数：

- 控制频率：`10-20 Hz`。
- FMM 重规划频率：`2-5 Hz`。
- 最大线速度：`0.2 m/s`。
- 最大角速度：`0.4 rad/s`。
- STG 到达半径：约 `0.1 m`，到达后等待下一次 FMM 更新，不能直接结束整个任务。

控制器输出标准 ROS 话题：

```text
/ieve/fmm_stg   geometry_msgs/PoseStamped，frame_id=odom
/ieve/fmm_stop  std_msgs/Bool
/odom 或 TF     提供 T_odom_base
        |
FmmCmdVelController
        |
/cmd_vel        geometry_msgs/Twist
```

### 1.4 当前项目的具体改造边界

建议保留 `_get_stg()`、FMM 可达性判断和目标选择逻辑，新增实机执行分支：

1. FMM 继续输出 `stg`、`stop` 和 `replan`。
2. ROS Bridge 将本地栅格 STG 转换成 `odom` 下的 `PoseStamped`。
3. 新增 `FmmCmdVelController`，基于实时 `T_odom_base` 闭环发布物理单位的 `/cmd_vel`。
4. 仿真模式继续调用 Habitat `super().step(action)`；实机模式禁止调用它，改为等待 `RealRobotEnv` 的下一帧真实 RGB-D、位姿和底盘反馈。
5. 使用 Mid-360 点云生成的障碍地图覆盖或补充 FMM 的障碍通道，并按照机器人真实外形膨胀障碍物。

实机控制还必须具备以下停车条件：

- Point-LIO 位姿、STG、点云或底盘反馈超时。
- Mid-360 前方安全区域出现近距离障碍物。
- FMM 返回最终目标到达或上层要求停止。
- ROS 节点异常退出、通信中断或人工急停触发。

因此，首版推荐方案是“现有 FMM + 新增 STG 闭环控制器 + `/cmd_vel` + 独立安全层”，而不是强制切换到 Nav2。

## 2. 总体结论

针对 Livox Mid-360，实机第一版采用 **Livox ROS Driver 2 + Point-LIO**。Mid-360 点云和内置 IMU 负责车辆 6DoF 定位，RGB-D 相机负责语义和高斯建图，使用 `direct_visual_lidar_calibration` 标定 LiDAR 与相机之间的刚性外参，再通过时间同步将雷达定位结果转换成相机曝光时刻的相机位姿。

第一版暂时不需要实现复杂的雷视紧耦合 SLAM。更稳妥的工程路线是：

1. 先把 Mid-360 + IMU 的激光惯性里程计调稳定。
2. 再完成雷达、相机和车体之间的外参标定。
3. 将 LIO 位姿插值到相机曝光时刻，作为 3DGS 的相机位姿。
4. 最后接入 IEVE 的二维语义地图、目标确认和车辆控制。

当前 IEVE 不能直接部署到实机，因为它仍然读取 Habitat 真值位姿：

- 二维语义地图使用绝对真值位姿：`main.py:1912`。
- 3DGS 接收相机位置和仿真四元数：`main.py:257`。
- RGB、深度和位姿都来自 Habitat：`envs/habitat/imagegoal_env.py:596`。
- 相机内参由 `hfov` 估算，并非真实标定值：`main.py:102`。

## 3. 推荐系统架构

```text
Mid-360 点云 + 内置 IMU
          |
       Point-LIO
          |
     T_odom_imu(t)
       /         \
T_odom_base    T_odom_camera
    |               |
IEVE/FMM 定位    RGB-D 语义与 3DGS
    |
车辆速度控制

Mid-360 注册点云 -> 2D 障碍地图/Nav2 costmap -> 碰撞保护

direct_visual_lidar_calibration（离线）
  -> ^lidar T_camera 静态外参
  -> 用于将 Point-LIO 位姿转换为相机位姿
```

在这套架构中：

- Point-LIO 提供连续、高频的局部 6DoF 位姿。
- 相机图像用于实例检测、语义识别、目标匹配和高斯建图。
- Mid-360 注册点云同时生成可靠的障碍地图和安全层。
- IEVE 负责高层探索、目标确认和目标点生成。
- FMM 或 Nav2 负责路径规划，底盘控制器负责执行速度指令。

### 3.1 Livox ROS Driver 2 与 Point-LIO 的区别

`livox_ros_driver2` 是硬件驱动和数据协议适配层，主要职责是：

- 与 Mid-360 通信。
- 解析并发布原始点云。
- 发布 Mid-360 内置 IMU 数据。
- 保留设备时间戳、单点时间和反射强度等字段。

驱动本身不会根据点云计算车辆位置，也不会输出可供导航使用的连续轨迹。

Point-LIO 是 HKU-MARS 提出的点级激光惯性里程计与增量点云建图项目。它的核心输入和输出是：

```text
输入：带单点时间的 LiDAR 点云 + IMU + LiDAR-IMU 外参
处理：IMU 状态传播 + 按单点时间处理 + 点级运动补偿 + 点到地图滤波更新
输出：连续 6DoF 位姿 + 去畸变/注册点云 + 增量局部点云地图
```

Point-LIO 不把一整帧扫描简单视为同一时刻的数据，而是利用每个 LiDAR 点的采样时间进行点级处理。IMU 负责高频状态传播，每个有效 LiDAR 点与增量地图形成几何约束，持续修正位置、姿态、速度和 IMU bias。该方法强调高带宽和高动态运动下的鲁棒性，也适合具有非重复扫描模式的 Mid-360。

Point-LIO 不负责以下工作：

- 不负责 Mid-360 硬件通信，前面仍需要 `livox_ros_driver2`。
- 默认不提供回环检测和全局重定位。
- 不提供相机-LiDAR 外参标定。
- 不生成实例语义、高斯地图或导航目标。
- 不负责路径规划和底盘控制。

Point-LIO 所说的“mapping”主要是为了点到地图匹配而维护的增量几何地图，不等于 IEVE 的语义地图、3DGS 地图或 Nav2 costmap。

## 4. Mid-360 雷达定位实现

### 4.1 驱动与数据采集

使用 `livox_ros_driver2` 发布 Mid-360 点云和 IMU 数据。必须保留每个激光点的时间戳，不能在进入 Point-LIO 前将点云转换成丢失点时间信息的普通 PCD。单点时间正是 Point-LIO 进行点级状态更新和运动补偿的关键输入。

建议至少记录以下 ROS 话题：

- Mid-360 原始点云。
- Mid-360 内置 IMU。
- RGB 图像。
- 深度图像。
- 相机 `CameraInfo`。
- Point-LIO 输出的里程计、注册点云和状态诊断信息。
- 底盘轮速里程计（如果有）。
- TF 和底盘控制指令。

正式接入 IEVE 前，先录制 rosbag，使用离线数据完成定位、同步和建图验证。

### 4.2 Point-LIO 安装、配置与输出确认

选择支持当前 ROS 版本、`livox_ros_driver2` 消息格式和 Mid-360 的 Point-LIO 版本，使用其 Mid-360 配置作为起点。不同分支对 Livox custom message、`PointCloud2` 和时间字段的支持可能不同，不能只根据 topic 名判断接入成功，必须重点核对：

- LiDAR 和 IMU topic 名称。
- 点云消息类型。
- Mid-360 内置 LiDAR-IMU 外参。
- 点时间单位。
- Point-LIO 是否实际读取并使用了单点时间字段。
- IMU 噪声参数。
- `blind` 近距离过滤范围。
- 点云降采样大小。
- 输出坐标系名称和 Odometry child frame。

建议按以下顺序验证：

1. 单独启动 `livox_ros_driver2`，确认原始点云和 IMU 时间戳单调、频率稳定。
2. 启动 Point-LIO，车辆保持静止完成重力方向和 IMU bias 初始化。
3. 先手推直线，再原地转动，最后走一个闭环。
4. 在 RViz 中同时观察原始点云、注册点云、轨迹和 TF。
5. 查清 Point-LIO 的 Odometry pose 表示的是 IMU、LiDAR 还是自定义 `body`，不能直接把任意 Odometry child frame 当成 `base_link`。

如果 Point-LIO 输出的是 `^odom T_imu`，则还需要通过静态外参计算：

```text
^odom T_lidar = ^odom T_imu * ^imu T_lidar
^odom T_base  = ^odom T_imu * ^imu T_base
```

只有明确输出 pose 对应的物理坐标系后，才能继续计算相机位姿。

开机后让车辆静止约 5 到 10 秒，完成重力方向和 IMU bias 初始化。传感器必须刚性安装，并尽量隔离电机和底盘的高频振动。

第一阶段先不要融合轮速里程计，单独验证 Point-LIO。Point-LIO 稳定后，如果长走廊或大面积平面场景仍出现几何退化，可以评估增加轮速线速度或非完整运动约束。不要在外部滤波器中重复融合已经被 Point-LIO 使用过的同一组 IMU 数据。

### 4.3 局部里程计与全局定位

Point-LIO 输出的是连续局部里程计，不等于固定地图中的全局定位。应根据任务模式选择：

- 每次任务从当前位置开始探索和建图：直接将 `odom` 作为 IEVE 世界原点。
- 需要在预建地图中启动：增加 Scan Context 或人工初值，并使用 ICP/NDT 完成初始扫描到地图配准，发布 `map -> odom`。
- 需要长距离闭环：增加回环检测和位姿图优化。

当前 IEVE 3DGS 将输入位姿视为固定历史位姿。闭环优化修改历史轨迹后，已有高斯不会自动随位姿图变形。因此第一版建议使用连续的 LIO `odom`，暂不启用会产生明显历史位姿修正的回环。

纯 ICP 或 NDT 不适合作为 Mid-360 的主要里程计，在走廊、白墙、玻璃和快速转弯场景中更容易退化。Point-LIO 的点级更新和高带宽输出适合快速转动和运动变化，但仍然依赖有效的几何结构、正确的单点时间、IMU 参数和 LiDAR-IMU 外参。

## 5. 雷达、相机与车体标定

### 5.1 坐标系设计

建议建立以下 TF 树：

```text
map -> odom -> base_link -> livox_frame
                         -> camera_link -> camera_optical_frame
```

各坐标系职责如下：

- `map`：存在全局重定位或回环时使用的全局坐标系。
- `odom`：连续、短期平滑的局部里程计坐标系。
- `base_link`：车辆控制与足迹坐标系。
- `livox_frame`：Mid-360 激光坐标系。
- `camera_link`：相机机械安装坐标系。
- `camera_optical_frame`：图像投影使用的光学坐标系。

TF 树中不能存在循环，也不能同时由多个节点发布同一对父子变换。

### 5.2 标定顺序与工具边界

推荐按以下顺序标定：

1. 相机内参和畸变参数。
2. RGB 与深度之间的对齐关系。
3. Mid-360 内置 IMU 到 LiDAR 的外参。
4. LiDAR 到 `base_link` 的外参。
5. LiDAR 到相机的 6DoF 外参。
6. 相机与 LiDAR 的时间偏移。

LiDAR-IMU 外参优先使用 Livox 官方提供的 Mid-360 内部标定参数。不要在基础系统尚未稳定时同时在线估计过多外参。

这里使用 `direct_visual_lidar_calibration` 完成第 5 项。它是一个无目标 LiDAR-相机外参标定项目，利用自然场景中的视觉信息、LiDAR 反射或几何结构优化相机与 LiDAR 之间的刚性变换，不要求在场景中摆放棋盘格或 AprilTag 标定板。

该项目负责的是 LiDAR 与相机之间的空间外参，不负责：

- 相机自身的内参和畸变标定。
- RGB 与深度相机之间的厂内标定或 depth-to-color 对齐。
- Mid-360 LiDAR 与内置 IMU 外参。
- `base_link` 与 Mid-360 之间的安装外参。
- LiDAR 与相机时间偏移，除非所使用版本明确启用了对应的时间标定功能。
- Point-LIO 定位、点云建图或车辆导航。

### 5.3 direct_visual_lidar_calibration 标定后得到什么

标定的核心结果是一个旋转和平移组成的 6DoF 刚性变换，通常保存为 4x4 齐次矩阵。例如工具输出：

```text
^camera T_lidar =
[ R_camera_lidar  t_camera_lidar ]
[       0 0 0              1     ]
```

它表示如何把 LiDAR 坐标系中的点转换到相机坐标系：

```text
p_camera = ^camera T_lidar * p_lidar
```

再结合相机内参，可以将 LiDAR 点投影到图像：

```text
[u, v, 1]^T ~ K * [x_camera / z_camera, y_camera / z_camera, 1]^T
```

标定正确后能够实现以下效果：

- 将 Mid-360 点云稳定投影到 RGB 图像，墙角、门框和物体轮廓基本对齐。
- 使用 RGB 给 LiDAR 点云着色。
- 将 LiDAR 距离投影成相机视角下的稀疏深度。
- 将 Point-LIO 的 LiDAR/IMU 位姿转换为相机位姿。
- 让相机语义、3DGS 和 LiDAR 障碍地图使用同一个空间坐标系。

本节公式中的 `camera` 应明确对应标定工具实际采用的相机坐标系。图像投影通常需要 `camera_optical_frame`；如果工具输出的是 `camera_link` 外参，还必须组合固定的 `camera_link -> camera_optical_frame` 变换。

标定不会让点云天然变成稠密深度，也不会自动完成实时数据融合。运行时仍然需要读取标定矩阵、查询 Point-LIO 位姿并按照时间戳计算 `T_odom_camera`。

由于 LiDAR 和相机视点不同，即使外参完全正确，物体遮挡边缘附近也可能出现少量投影不一致。运动场景还会受到时间偏移、滚动快门和点云运动畸变影响。因此不能只观察一张静止图片判断标定质量。

### 5.4 direct_visual_lidar_calibration 实施步骤

#### 步骤一：冻结运行时传感器模式

在标定前固定以下参数，之后运行 IEVE 时保持一致：

- RGB 图像分辨率、帧率、裁剪和缩放方式。
- 图像是否已经去畸变。
- 使用原始 RGB 还是 rectified RGB。
- 对应图像的 `CameraInfo.K` 和畸变模型。
- Mid-360 点云消息类型、时间戳来源和点云坐标系。

如果标定使用原始图像，运行时却使用裁剪、缩放或去畸变后的图像，必须同步更新相机内参；不能直接复用原始 `K`。

#### 步骤二：准备初始外参

使用机械图、尺量或现有静态 TF 给出合理的 LiDAR-相机初始旋转和平移。`direct_visual_lidar_calibration` 属于非线性优化方法，初值方向反了、轴定义错了或误差过大时，可能收敛到错误局部最优。

初值准备时应明确：

- LiDAR frame 名称及轴方向。
- 相机使用 `camera_link` 还是 `camera_optical_frame`。
- 矩阵是 `T_camera_lidar` 还是 `T_lidar_camera`。
- 平移单位是米还是毫米。
- 四元数顺序是 `xyzw` 还是 `wxyz`。

#### 步骤三：采集无目标标定数据

选择同时具有纹理、反射变化和三维几何边缘的静态环境，例如门框、墙角、桌椅和立柱。采集时应满足：

- LiDAR 与相机有足够的公共视场。
- 覆盖图像中心、四周以及近、中、远距离。
- 覆盖多个朝向，不能只对着一面平墙。
- 避免大面积白墙、玻璃、镜面、强阳光和大量运动人员。
- 固定自动曝光设置，或避免采集期间出现剧烈曝光变化。
- 传感器安装支架保持刚性，采集后不能再移动安装位置。

Mid-360 是非重复扫描 LiDAR。如果为了增加点密度而累计多帧点云，传感器运动时必须先使用 Point-LIO 或工具自身支持的轨迹进行去畸变和运动补偿，不能直接叠加原始扫描。

#### 步骤四：运行标定优化

向 `direct_visual_lidar_calibration` 提供与运行时一致的图像、`CameraInfo`、Mid-360 点云和初始外参。具体命令和配置字段以当前安装版本的 README 为准，不在 IEVE 中硬编码工具版本相关参数。

建议至少使用两到三组独立数据重复标定，并保存：

- 4x4 外参矩阵。
- 输入和输出 frame 名称。
- 矩阵变换方向。
- 相机分辨率、内参和畸变状态。
- 数据集名称、标定时间和工具版本。
- 优化初值、最终代价和可视化结果。

优化代价下降只表示目标函数收敛，不代表外参一定正确，必须继续做独立投影验证。

#### 步骤五：确认矩阵方向并发布 TF

如果工具输出的是 `^camera T_lidar`，而 TF 链或代码需要 `^lidar T_camera`，则必须取逆：

```text
^lidar T_camera = inverse(^camera T_lidar)
```

假设已经知道 `^base T_lidar`，则：

```text
^base T_camera = ^base T_lidar * ^lidar T_camera
```

将最终外参写入 URDF/xacro 或唯一的静态 TF publisher。不要同时由标定节点、URDF 和另一个静态节点重复发布同一对 TF。

#### 步骤六：使用独立数据验证

不要用参与优化的同一小段数据作为唯一验收依据。使用独立 rosbag 验证：

- 静止场景下，近、中、远距离的墙角、门框和桌边是否同时对齐。
- 图像中心和四周是否没有相反方向的系统误差。
- 车辆缓慢平移和原地旋转时是否仍然对齐。
- 不同数据重复标定得到的外参是否接近。

建议将多次标定结果的平移和旋转差异作为稳定性指标。若不同数据得到的结果差异达到数厘米或数度，不能直接取平均，应重新检查相机模式、初值、场景质量、点云去畸变和时间同步。

标定完成后必须将点云投影到图像上检查：

- 墙角、桌边和门框是否重合。
- 图像中心和边缘是否都没有系统偏移。
- 不同距离下是否都能保持一致。
- 车辆静止和缓慢旋转时是否仍能稳定重合。

如果静止时重合、运动时错位，通常应优先检查时间同步，而不是重新调整空间外参。

## 6. 时间同步与位姿插值

定义 `^A T_B` 为将 B 坐标系中的点变换到 A 坐标系，则相机位姿为：

```text
^odom T_camera(t_c)
  = ^odom T_imu(t_c) * ^imu T_lidar * ^lidar T_camera
```

或者在 Point-LIO 已输出 LiDAR 位姿时：

```text
^odom T_camera(t_c)
  = ^odom T_lidar(t_c) * ^lidar T_camera
```

其中 `t_c` 应当是相机曝光中点对应的时间戳。

必须在相机曝光时刻对 Point-LIO 位姿进行插值：

- 平移使用线性插值。
- 旋转使用四元数 SLERP。
- 查询时间应包含标定得到的相机-LiDAR 时间偏移。

只使用距离相机帧最近的一帧 Odometry，会在转弯时造成目标投影偏移和高斯重影。例如车辆以 `0.5 m/s` 运动时，`20 ms` 时间误差会产生约 `1 cm` 平移误差；快速旋转时姿态误差通常更明显。

建议优先使用相机和 Mid-360 支持的硬件同步或 PTP。无法硬件同步时，应离线估计时间偏移，并将相机到有效位姿的时间误差控制在约 5 到 10 ms 内。ROS `ApproximateTime` 只能帮助配对消息，不能代替硬件同步和位姿插值。

## 7. 相机与高斯建图要求

当前 IEVE 的语义地图和 3DGS 都依赖 RGB-D：

- `gaussian_mapping/configs/base_config.yaml` 中使用 `sensor_type: rgbd`。
- `main.py:257-271` 将 RGB、深度和位姿传给 3DGS。
- `model.py:221-227` 使用深度生成点云。

如果只有普通 RGB 相机，将 Mid-360 点云投影到图像后只能得到稀疏深度，通常不足以直接支持当前的语义地图与 3DGS。首版建议使用 RGB-D 相机或双目深度相机，并保证：

- RGB 和深度已经对齐。
- 深度单位统一为米。
- 无效深度统一为 0。
- RGB、深度、实例 mask 和相机位姿严格对应同一时刻。
- 图像输入分辨率与相机内参完全匹配。

当前 `build_3dgs_camera_params()` 通过 `hfov` 推导焦距，并假设 `fx == fy`、主点位于图像中心。实机应从校正后图像对应的 `CameraInfo.K` 读取真实的 `fx`、`fy`、`cx` 和 `cy`。

## 8. IEVE 实机接入方案

### 8.1 新增 ROS 实机桥接层

建议新增一个 `RealRobotEnv` 或 ROS Bridge，生成与当前 Habitat `info` 相同或兼容的数据：

- `raw_rgb`：校正并对齐后的 RGB 图像。
- `raw_depth`：单位为米的稠密深度图，无效值为 0。
- `ieve_gt_pose`：名称后续可重构，首版填入 LIO 的 `[x, y, yaw]`。
- `agent_position`：车体在 IEVE 世界坐标系中的位置。
- `agent_rotation_xyzw`：车体世界姿态。
- `camera_position`：相机光心的世界位置。
- `camera_rotation_xyzw`：相机完整世界姿态。
- 原始传感器时间戳和用于查询位姿的校正时间戳。

当前 `update_3dgs_mappers()` 使用的是“相机位置 + 车体旋转”。实机相机存在完整安装外参，因此应优先读取 `camera_rotation_xyzw`，仅在缺少该字段时回退到 `agent_rotation_xyzw`。

### 8.2 ROS 与 IEVE 坐标转换

ROS 通常使用：

- `x` 向前。
- `y` 向左。
- `z` 向上。

当前 IEVE/MinAV 延续 Habitat 坐标约定，将 `x-z` 作为水平面、`y` 作为竖直方向。例如 `ieve_minav/manager.py:10628` 使用 `x-z` 距离。

为了减少对 MinAV 和高斯导航逻辑的改动，建议在实机桥接层将 ROS 坐标转换成 IEVE 内部坐标：

```text
p_ieve = [-y_ros, z_ros, -x_ros]
```

对应旋转矩阵为：

```text
R_ieve_from_ros =
[[ 0, -1,  0],
 [ 0,  0,  1],
 [-1,  0,  0]]
```

相机和车体姿态也必须乘同一个旋转矩阵，不能只交换位置坐标。

用于 IEVE 二维规划的 `ieve_gt_pose` 可以继续使用 ROS 平面定义 `[x_ros, y_ros, yaw_ros]`。当前 `ieve_full_pose_from_gt()` 会减去 episode 起点并将车辆放置在地图中心，因此适合使用任意 LIO `odom` 原点。

### 8.3 3DGS 位姿接口

当前 `gaussian_mapping/mapper/realtime_3dgs.py:457-468` 在四元数转换中加入了固定的 OpenCV/OpenGL 轴翻转。实机接入时，建议让 3DGS 直接接收显式的 `T_world_camera_optical` 4x4 矩阵，并在一个有单元测试的函数中完成 ROS optical、OpenCV 和 OpenGL 的唯一一次转换。

不要在多个 ROS callback、环境接口和 3DGS mapper 中分别交换坐标轴，否则很容易产生镜像、旋转 90 度或目标投影系统偏移。

### 8.4 障碍地图和车辆安全

相机高斯地图适合语义表示和目标确认，但不应成为车辆唯一的碰撞保护来源。建议同时使用 Mid-360 点云构建二维或三维局部 costmap：

- 高度过滤后投影成二维占据栅格。
- 根据车辆外形膨胀障碍物。
- 对玻璃、近距离盲区和车体自反射点进行过滤。
- 在 IEVE 速度输出后增加独立的紧急停车和限速层。

首版可以保留 IEVE FMM 作为研究算法的规划器，同时用 LiDAR costmap 覆盖或补充 IEVE 的障碍通道。长期可以让 IEVE 只输出目标点，由 Nav2 完成局部规划和速度控制。

## 9. 推荐调试步骤

### 阶段一：传感器和 Point-LIO

1. 单独启动 `livox_ros_driver2`，检查点云、IMU 频率和时间戳是否连续。
2. 确认点云保留 Mid-360 单点时间信息后再启动 Point-LIO。
3. 静止初始化后观察 Point-LIO 位姿是否明显漂移。
4. 手推或低速运行，检查注册点云是否有双墙和拖影。
5. 完成 30 到 50 米闭环，记录终点平移和航向误差。
6. 明确 Point-LIO 输出 pose 对应的物理 frame，并推导 `T_odom_base` 和 `T_odom_lidar`。

建议首阶段验收目标：闭环返回误差约小于 `0.2 m / 1 deg`，静止时位姿没有持续单向漂移。

### 阶段二：标定和同步

1. 固定相机分辨率、图像去畸变方式和运行时 `CameraInfo`。
2. 完成相机内参和 RGB-D 对齐，记录最终使用的 `K`。
3. 检查 Mid-360 点云单点时间和 Point-LIO 去畸变结果。
4. 测量并记录 `direct_visual_lidar_calibration` 所需的合理初始外参。
5. 在纹理、反射和几何结构丰富的静态环境采集多组无目标标定数据。
6. 使用 `direct_visual_lidar_calibration` 分别优化多组数据，保存 4x4 外参、方向和元数据。
7. 将 LiDAR 点投影到未参与优化的图像，检查近、中、远距离和图像四周。
8. 将确认后的外参写入 URDF/xacro 或唯一静态 TF publisher。
9. 缓慢平移和旋转车辆，估计相机-LiDAR 时间偏移。
10. 将 Point-LIO 位姿插值到相机曝光时刻，再次验证动态投影。

### 阶段三：离线高斯建图

1. 使用 rosbag 离线生成同步的 RGB-D 和相机位姿。
2. 关闭 IEVE 导航，只运行 3DGS 建图。
3. 检查墙面、家具边缘和目标实例是否重影。
4. 检查相机渲染视角与真实图像是否一致。
5. 检查实例 mask、深度和相机位姿是否严格同帧。

### 阶段四：二维地图和目标导航

1. 将 LIO `[x, y, yaw]` 接入 IEVE 的绝对位姿接口。
2. 对比车辆真实位置、IEVE 栅格位置和规划器起点。
3. 接入 Mid-360 障碍地图。
4. 只执行定点导航，不运行目标确认。
5. 验证转向符号、速度单位和停止逻辑。

### 阶段五：完整实机运行

1. 开启实例检测和目标匹配。
2. 开启 3DGS 目标语义更新。
3. 开启 IEVE 探索、确认和利用阶段。
4. 初始速度限制在 `0.2 m/s` 以内。
5. 独立记录 RGB-D、LIO、TF、IEVE 状态、目标点和控制指令。

## 10. 常见问题定位

### 静止时点云和图像对不上

优先检查：

- 相机内参和畸变。
- LiDAR-相机外参方向是否取反。
- `camera_link` 与 `camera_optical_frame` 是否混用。
- 投影使用的图像是否经过裁剪或缩放。
- `direct_visual_lidar_calibration` 使用的相机模式是否与运行时一致。
- 标定矩阵方向是否需要取逆，平移单位是否正确。
- 初始外参是否过差，优化是否落入错误局部最优。
- 标定场景是否缺少纹理、反射变化和三维结构。

### 静止时正常，运动时错位

优先检查：

- 相机与 LiDAR 时间偏移。
- 是否使用相机曝光中点时间。
- 是否对 LIO 位姿进行插值。
- 点云是否保留每个点的时间信息。
- 为增加 Mid-360 点密度而累计点云时，是否进行了运动补偿。

### 二维地图旋转或镜像

优先检查：

- ROS `x-y-z` 与 IEVE `x-z/y-up` 转换。
- yaw 正方向。
- `agent_position` 与 `ieve_gt_pose` 是否使用了不同世界原点。
- 地图 row/column 与世界 x/y 是否混淆。

### 3DGS 出现双墙或目标高斯重影

优先检查：

- 相机位姿是否为相机光心位姿，而不是车体位姿。
- 相机旋转是否包含完整安装外参。
- RGB、深度、mask 和位姿是否同帧。
- 深度单位是否为米。
- 是否在闭环修正后继续向未变形的旧高斯地图添加数据。

## 11. 建议的实施优先级

推荐按照以下优先级推进：

1. 使用 Livox ROS Driver 2 得到时间字段完整且稳定的 Mid-360 点云和 IMU。
2. 使用 Point-LIO 得到稳定、高频的局部 6DoF 位姿、去畸变点云和注册点云。
3. 固定相机模式并验证真实内参、畸变和 RGB-D 对齐。
4. 使用 `direct_visual_lidar_calibration` 得到可重复、方向明确的 LiDAR-相机 6DoF 外参。
5. 完成 LiDAR-相机时间偏移估计和相机曝光时刻位姿插值。
6. 使用 rosbag 验证离线 RGB-D + LIO 高斯建图。
7. 使用 LIO 二维位姿替换 Habitat 真值位姿。
8. 将 Mid-360 障碍地图接入 FMM 或 Nav2。
9. 接入实机底盘速度控制和独立安全层。
10. 运行完整 IEVE 高斯实例图像导航。
11. 最后再考虑轮速紧耦合、全局重定位和回环优化。

这一路线能够把问题拆分为定位、标定、时间同步、建图、规划和控制六个可独立验收的模块，避免在完整系统运行失败时无法判断问题来自哪一层。
