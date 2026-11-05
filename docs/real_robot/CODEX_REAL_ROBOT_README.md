# 实机 Codex 交接 README

日期：2026-11-05。请把本文件作为接手调试的上下文；完整任务清单见 [Real_Robot_Quantitative_Experiment_Checklist_20261105.md](Real_Robot_Quantitative_Experiment_Checklist_20261105.md)。

## 已知事实

- 机器人上已经有**能运行单目标demo的OSGSNav代码**。先检查并归档这份实际代码，不能用本仓库Habitat入口覆盖它。
- 本仓库提供Ours两目标/五目标V5参考链路；`baselines/unigoal_m/`提供UniGoal-M两目标/五目标适配层及上游锁定信息。
- UniGoal-M当前是Habitat适配，不是已跑通的实机实现。上游`real_world_env.py`有TODO及零值观测；`--real_world`参数不表示后端已完成。
- 服务器上的历史仿真和runtime数值不能作为实机测量结果。

## 最终目标和顺序

1. 归档机器人上的Ours demo版本、启动命令、配置、标定、依赖和一条完整日志。
2. 确认UniGoal固定依赖可安装，接入相同真实RGB-D/外部位姿/底盘执行接口，先完成单目标。
3. 先固定共同评测协议和自动日志，再做3–5任务pilot，冻结版本后开展 **Ours vs UniGoal单目标定量实验**。
4. 移植两目标切换，覆盖成功/失败继续、不同类及同类不同实例。
5. 做2–3条五目标pilot，核验4次切换和长期内存，再开展 **Ours vs UniGoal-M五目标定量实验**。
6. 保存全量失败与异常，按共同task/sequence配对统计；不择优替换结果。

建议正式规模：单目标20任务×2方法×2重复；多目标10条五目标序列×2方法×2重复。它们是资源规划建议，不能当成已完成或统计功效保证；开始正式采集前由实验负责人冻结实际规模。

## 首轮接手时必须先做的检查

- 从本机已有目录、启动脚本、进程/日志找到实际demo入口，不凭本文件猜路径。记录commit及dirty diff或文件hash，先保留可运行版本。
- 查明ROS版本、相机/深度编码、曝光时间戳、内参、相机—底盘—LIO外参、外部定位topic/坐标系、现有控制和STOP接口。
- 明确什么已真实接入、什么仍是stub。禁止用固定零观测/零pose通过接口测试。
- 建立独立开发分支/配置/结果目录；修改前记录基线，不覆盖原demo、历史配置、标定和结果。
- 先用静止采样或录制数据核对投影、深度尺度、地图方向、时间对齐，再接闭环导航。沿用现场已验证的底盘保护与现场运动授权范围。
- 第一个可交付结果应是“本机现状、实际入口、复现命令、缺口、下一项最小改动”，随后落实该改动，不停留在泛泛计划。

## 两方法共同契约

输入：当前目标RGB、同类型实时RGB-D、CameraInfo、曝光时刻外部相机/底盘位姿。目标无depth；禁止额外提供GT目标位置、实例mask等。类别要么两者都提供相同任务类别，要么按冻结的识别协议获取，不能一方给oracle、另一方自己识别。

执行：两者采用相同底盘限速和执行/反馈层。Habitat归一化速度不能直接作为m/s或rad/s。保留两方法各自高层规划；UniGoal保留所需LLM/VLM，固定模型digest/量化/提示词/重试策略，任务时间包含全部请求和网络等待。

任务：单目标每次清空地图；多目标仅序列开始清图，目标间保留各自历史地图和实际位置，前目标失败仍继续，policy不能提前看到未来目标图像。T2以后两方法实际起点不同属于闭环结果，不人工搬到相同位置。

预算与成功：正式采集前冻结共同每目标墙钟预算，可加距离上限；原生steps只做辅助。成功由指定物理实例、预定到达区域与自主停止共同判断，不使用internal localization标志替代外部成功。

SPL：仅在独立参考最短可行路径可靠时计算，从各目标实际起点测量；不能用方法自己的预测地图定义GT。否则报告SR、实际距离、任务时间，并说明未报SPL。

## 目标切换验收

Ours应保留Gaussian/occupancy/instance/keyframe状态；UniGoal-M应保留BEV、collision/visited与scene graph。两者清理query候选、拒绝/匹配、临时导航目标及目标条件评分。

每次切换保存query ID、goal图hash、类别、pose与地图规模。特别检查：

- 同类不同物理实例是否错误继承“已确认/已定位”；
- Ours旧semantic scalar、mask、lock是否残留；
- UniGoal旧graph评分/子图是否清理，而历史节点是否保留；
- 预算是否按目标重置、失败是否继续、4次切换是否完整；
- RSS、GPU、Gaussian/keyframe/graph数量是否持续危险增长。

发现算法问题先记录、提交独立修复和回归结果；不静默修复后将新旧结果混统。危险内存增长时停止保留日志，不无限提高配额或偷偷清空Shared地图。

## 日志与验收产物

每次run至少保存：源码hash、resolved config、模型/标定hash、task/sequence/repetition、goal图hash、初始与逐目标实际起点、预算、轨迹、控制反馈、事件时间戳、LLM调用、RSS/GPU、成功与终止原因、人工介入和退出码。

候选检索、内部匹配/定位完成、正确实例验证、最终导航成功必须分开。Step-0检索必须发生在针对新query的运动/新观测更新之前；不能满足时改名并记录实际顺序。

调试pilot和正式结果分开。基础设施重试保留首次记录、原因与完整版本，不把异常默认为导航失败，也不删掉方法失败。多目标统计/置信区间以sequence为聚类单位，不把同序列5个请求当独立样本。

每完成一阶段交付：实际改动与diff、可运行命令、验证结果/原日志、尚未解决的问题、下一阶段验收条件。进度回复简短，细节写文件；用户希望减少token消耗。

## 关键参考路径

- Ours仿真：`five_goal_benchmark_20261026/runtime/main_multi_iin_frontier_v5.py`
- UniGoal-M：`baselines/unigoal_m/README.md`、`k2/`、`k5/`、`upstream.lock.json`
- UniGoal预检：`python baselines/unigoal_m/run.py --unigoal-root <checkout> --goals 5 --output <fresh-dir>`（不加`--execute`仅预检）
- 实机接口：`docs/habitat_to_real_robot_interface_summary.md`
- 同步与停走：`docs/real_robot/real_robot_30hz_stop_go_implementation.md`
- 完整清单：本目录`Real_Robot_Quantitative_Experiment_Checklist_20261105.md`

## 可直接粘贴给实机Codex的任务

> 阅读本README及完整清单。先检查这台机器人上已跑通的OSGSNav单目标demo，归档真实源码版本、配置、标定、依赖和启动命令；不要覆盖它。随后核对本仓库独立UniGoal-M适配层及固定依赖，找出真实RGB-D/位姿/控制与评测接口的缺口，优先完成UniGoal单目标实机接入与共同日志。按“单目标pilot→正式单目标对照→两目标切换→五目标pilot→正式多目标对照”推进。只报告实际验证的能力，保留所有失败，不混用源码版本，不把内部定位当实机成功。先落实可独立完成的准备与验证，进度简短、详细证据写文件。
