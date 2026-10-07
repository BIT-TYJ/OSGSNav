"""
Exploration Planner (EP) 模块

根据 LOCALIZATION_INSTRUCTION.md 的设计:
EP模块当fem触发之后，将控制机器人前往记录下来的最高匹配度渲染器位姿。
由habitat系统判定是否到达目标，输出成功或失败，无论成败都可以终止此次导航运行。

但是由于 run.py 已经集成了比较完善的目标确认后导航的逻辑，
所以该文件只作为接口，最终使用 run.py 中已经存在的导航模块。
"""

import logging
import numpy as np
from typing import Dict, Any, Optional


class ExplorationPlanner:
    """
    探索规划模块 - 作为定位系统与外部导航系统的接口
    
    实际的导航逻辑由 run.py 中的 PathPlanner 和导航循环处理。
    此类仅提供接口和状态管理。
    """
    
    def __init__(self, habitat_sim=None):
        """
        初始化EP模块
        
        Args:
            habitat_sim: Habitat仿真环境引用（可选，主要用于未来扩展）
        """
        self.logger = logging.getLogger(__name__)
        self.habitat_sim = habitat_sim
        
        # 目标位姿（由FEM模块设置）
        self.target_pose = None
        self.target_position = None
        self.target_yaw = None
        
        # 导航状态
        self.navigation_started = False
        self.navigation_completed = False
        self.navigation_success = False
        
        self.logger.info("ExplorationPlanner initialized (interface mode)")
    
    def set_target_pose(self, pose: np.ndarray, position: Optional[np.ndarray] = None, 
                        yaw: Optional[float] = None):
        """
        设置导航目标位姿
        
        Args:
            pose: 4x4变换矩阵 (w2c格式)
            position: 目标位置 [x, y, z]（可选，从pose提取）
            yaw: 目标朝向角度（可选，从pose提取）
        """
        self.target_pose = pose
        
        if position is not None:
            self.target_position = position
        elif pose is not None:
            # 从w2c矩阵提取位置
            c2w = np.linalg.inv(pose)
            self.target_position = c2w[:3, 3].copy()
        
        if yaw is not None:
            self.target_yaw = yaw
        elif pose is not None:
            # 从w2c矩阵提取朝向
            c2w = np.linalg.inv(pose)
            forward = -c2w[:3, 2]  # 相机 -Z 方向
            self.target_yaw = np.rad2deg(np.arctan2(forward[0], -forward[2]))
        
        self.logger.info(f"EP target set: position={self.target_position}, yaw={self.target_yaw}°")
    
    def navigate_to_pose(self, pose: np.ndarray) -> Dict[str, Any]:
        """
        导航到目标位姿（接口方法）
        
        注意：实际导航由 run.py 的主循环处理。
        此方法仅用于设置目标并返回状态信息。
        
        Args:
            pose: 目标位姿 (4x4 w2c矩阵)
            
        Returns:
            导航结果字典，包含:
            - success: 是否成功（这里返回False，等待外部导航系统完成）
            - message: 状态消息
            - target_position: 目标位置
            - target_yaw: 目标朝向
        """
        self.set_target_pose(pose)
        self.navigation_started = True
        
        # 返回等待状态，实际导航由 run.py 处理
        return {
            'success': False,  # 导航尚未完成
            'pending': True,   # 标记为等待外部导航
            'message': 'Navigation target set, waiting for external navigation system',
            'target_position': self.target_position,
            'target_yaw': self.target_yaw,
            'pose': pose
        }
    
    def get_navigation_target(self) -> Optional[Dict[str, Any]]:
        """
        获取当前导航目标（供外部导航系统使用）
        
        Returns:
            目标信息字典，如果没有目标则返回None
        """
        if self.target_position is None:
            return None
        
        return {
            'position': self.target_position,
            'yaw': self.target_yaw,
            'pose': self.target_pose
        }
    
    def mark_navigation_complete(self, success: bool, final_position: Optional[np.ndarray] = None):
        """
        标记导航完成（由外部导航系统调用）
        
        Args:
            success: 是否成功到达目标
            final_position: 最终位置
        """
        self.navigation_completed = True
        self.navigation_success = success
        
        if final_position is not None and self.target_position is not None:
            distance = np.linalg.norm(final_position - self.target_position)
            self.logger.info(f"Navigation completed: success={success}, distance={distance:.3f}m")
        else:
            self.logger.info(f"Navigation completed: success={success}")
    
    def reset(self):
        """重置EP模块状态"""
        self.target_pose = None
        self.target_position = None
        self.target_yaw = None
        self.navigation_started = False
        self.navigation_completed = False
        self.navigation_success = False
        self.logger.info("ExplorationPlanner reset")
