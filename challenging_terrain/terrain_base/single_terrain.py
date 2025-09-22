"""
🏞️ 人形机器人地形挑战基准测试 - 地形类型与评估场景完整对应表
=================================================================

本文件包含10种不同的地形类型生成函数，每种地形对应特定的评估场景和能力测试目标。

📊 地形与评估场景完整对应关系：

🌟 Simple Terrain（简单地形）
├── flat()     - 平坦地形：基础行走能力验证
└── wave()     - 波浪地形：轻微起伏适应能力

🏃 Normal Terrain（正常地形）  
├── hurdle()   - 跨栏障碍：连续跨越和节奏控制
├── slope()    - 斜坡地形：坡度行走和重心调节
└── uneven()   - 不平整地面：复杂表面适应能力

💪 Hard Terrain（困难地形）
├── parkour()  - 跨越障碍：跳跃到石头平台的精确控制
└── gap()      - 间隙跳跃：深坑跳跃的爆发力和着陆控制

🎯 Challenging Terrain（挑战地形）
├── bridge()   - 窄桥行走：高精度平衡和路径跟踪
└── plot()     - 精确踩点：极小空间的精确落脚控制

🔬 专项评估地形：

🔄 Robustness Evaluation（鲁棒性评估）
└── stair()    - 阶梯地形：扩展阶梯数量（5-10个 → 20+个）
               测试长距离连续爬升的稳定性和耐久性

⚡ Extreme Evaluation（极限评估）  
└── stair()    - 阶梯地形：增加阶梯高度（0.08-0.2m → 更高）
               探索机器人运动能力的物理上限

🔄 Generalization Evaluation（泛化性评估）
└── 组合地形   - 多种地形类型的动态组合
               测试不同地形间的平滑过渡和技能泛化

🎯 评估目标总结：
- Simple → Normal → Hard → Challenging：渐进式难度递增
- Robustness：持续性和稳定性测试
- Extreme：能力上限探索
- Generalization：真实世界适应性验证

每种地形都包含详细的目标点设置逻辑，确保机器人有明确的导航目标和奖励信号。
"""
import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from .config import terrain_config
from scipy import ndimage
from pydelatin import Delatin
import pyfqmr
from scipy.ndimage import binary_dilation
import math

class single_terrain:
    """
    单一地形类型生成器 - 机器人地形挑战基准测试
    
    这个类包含了多种地形类型的生成方法，用于评估人形机器人在不同环境下的运动能力。
    每种地形都对应特定的评估场景和挑战类型。
    
    地形类型与评估场景对应关系：
    ====================================
    
    🏃 基础运动能力评估：
    - flat(): 平坦地形 → Simple Terrain（简单地形）
    - wave(): 波浪地形 → Simple Terrain（增加轻微起伏）
    
    🧗 垂直运动能力评估：
    - stair(): 阶梯地形 → Robustness Evaluation（鲁棒性评估）& Extreme Evaluation（极限评估）
    - slope(): 斜坡地形 → Normal Terrain（正常地形）
    
    🦘 跳跃和跨越能力评估：
    - parkour(): 跨越障碍 → Hard Terrain（困难地形）
    - gap(): 间隙跳跃 → Hard Terrain（困难地形）
    - hurdle(): 跨栏障碍 → Normal Terrain（正常地形）
    
    ⚖️ 平衡和精确控制评估：
    - bridge(): 窄桥行走 → Challenging Terrain（挑战地形）
    - plot(): 精确踩点 → Challenging Terrain（挑战地形）
    
    🌊 复杂表面适应评估：
    - uneven(): 不平整地面 → Normal Terrain（正常地形）
    
    🔄 组合地形评估：
    - 多种地形组合 → Generalization Evaluation（泛化性评估）
    """
    def __init__(self, cfg: terrain_config) -> None:
        self.cfg = cfg
    

    def parkour(terrain, 
            length_x=18.,
            length_y=4.,
            num_goals=6, 
            start_x=0,
            start_y=0,
            platform_size=2.5, 
            difficulty=0.5,
            x_range=[0.5, 1.0],
            y_range=[0.3, 0.4],
            stone_len_range=[0.8, 1.0],
            stone_width_range=[0.6, 0.8],
            incline_height=0.1,
            pit_depth=[0.5, 1.]):
        """
        生成跨越式障碍地形（Parkour）并设置导航目标点
        
        这个函数创建一个包含石头平台和深坑的挑战性地形，机器人需要跳跃或跨越
        石头来穿越地形。目标点被战略性地放置在每个石头平台上，引导机器人
        沿着安全路径前进。
        
        目标点设置策略：
        1. 第一个目标：起始平台边缘，准备跳跃
        2. 中间目标：每个石头平台的中心，确保安全落脚点
        3. 最后目标：终点位置，完成穿越
        
        Args:
            terrain: 地形对象
            length_x: 地形长度（米）
            length_y: 地形宽度（米）
            num_goals: 目标点数量
            difficulty: 难度系数 [0,1]，影响间隙大小和石头尺寸
            x_range: X方向间隙范围（米）
            y_range: Y方向间隙范围（米）
            stone_len_range: 石头长度范围（米）
            stone_width_range: 石头宽度范围（米）
        """
    
        # ===== 目标点数组初始化 =====
        goals = np.zeros((num_goals, 2))  # 创建目标点数组 [num_goals, 2(x,y)]
        
        # 随机生成深坑深度
        pit_depth_val = np.random.uniform(pit_depth[0], pit_depth[1])  # 0.5-1.0米深
        pit_depth_grid = -round(pit_depth_val / terrain.vertical_scale)  # 转换为网格单位（负值表示向下）
        
        # 获取地形缩放参数
        h_scale = terrain.horizontal_scale  # 水平缩放（米/网格）
        v_scale = terrain.vertical_scale    # 垂直缩放（米/网格）
    
        # 将物理尺寸转换为网格坐标
        length_y_grid = round(length_y / h_scale)  # 地形宽度（网格）
        mid_y = length_y_grid // 2                 # 地形中线Y坐标

        length_x_grid = round(length_x / h_scale)  # 地形长度（网格）
        
        # 根据难度系数计算障碍物参数（难度越高，石头越小，间隙越大）
        stone_len = round(((stone_len_range[0] - stone_len_range[1]) * difficulty + stone_len_range[1]) / h_scale)    # 石头长度
        stone_width = round(((stone_width_range[0] - stone_width_range[1]) * difficulty + stone_width_range[1]) / h_scale)  # 石头宽度
        gap_x = round(((x_range[1] - x_range[0]) * difficulty + x_range[0]) / h_scale)  # X方向间隙
        gap_y = round(((y_range[1] - y_range[0]) * difficulty + y_range[0]) / h_scale)  # Y方向间隙
        
        platform_size_grid = int(round(platform_size / h_scale))  # 起始平台大小
        incline_height_grid = int(round(incline_height / v_scale))  # 石头倾斜高度
        
        # ===== 创建深坑地形 =====
        # 在整个区域挖一个大坑，后续会在上面放置石头平台
        terrain.height_field_raw[start_x+platform_size_grid:start_x + length_x_grid, start_y:start_y+length_y_grid*2] = pit_depth_grid
        
        # ===== 目标点设置开始 =====
        
        # 计算第一个石头的位置
        dis_x = start_x + platform_size_grid - gap_x + stone_len // 2
        
        # 🎯 目标点1：起始平台的边缘，准备跳跃到第一个石头
        goals[0] = [start_x + platform_size_grid - stone_len // 2, start_y + mid_y]
        
        # 随机选择石头的左右摆放模式（增加路径多样性）
        left_right_flag = np.random.randint(0, 2)  # 0或1，决定第一个石头在左侧还是右侧
        
        # ===== 生成中间石头和对应的目标点 =====
        for i in range(num_goals - 2):  # 排除第一个和最后一个目标点
            dis_x += gap_x  # 下一个石头的X位置
            
            # 计算石头的Y位置（左右交替摆放）
            pos_neg = 2 * (left_right_flag - 0.5)  # 转换为 +1 或 -1
            dis_y = mid_y + pos_neg * gap_y         # 在中线上下偏移
            
            # 计算石头在网格中的边界
            x_start = int(dis_x - stone_len // 2)
            x_end = x_start + stone_len
            y_start = int(dis_y - stone_width // 2)
            y_end = y_start + stone_width
            
            # 创建石头表面的倾斜效果（增加挑战性）
            heights = np.tile(np.linspace(-incline_height_grid, incline_height_grid, stone_width),(stone_len, 1)) * pos_neg
            heights = heights.astype(int)
            
            # 边界检查，防止超出地形范围
            if x_end > terrain.height_field_raw.shape[0]:
                x_end = terrain.height_field_raw.shape[0]
            if y_end > terrain.height_field_raw.shape[1]:
                y_end = terrain.height_field_raw.shape[1]
    
            # 在地形上放置石头
            actual_height = heights[:x_end - x_start, :y_end - y_start]
            terrain.height_field_raw[x_start:x_end, y_start:y_end] = actual_height
            
            # 🎯 目标点i+1：放置在当前石头的中心位置
            # 这确保机器人瞄准石头的安全中心区域，而不是边缘
            goals[i + 1] = [dis_x, dis_y]
            
            # 切换左右标志，下一个石头放在相反一侧（创建之字形路径）
            left_right_flag = 1 - left_right_flag
        
        # ===== 设置最后一个目标点 =====
        final_dis_x = dis_x + gap_x  # 最后一个目标点的X位置
        
        # 🎯 最后一个目标点：回到中线，表示成功穿越所有障碍
        goals[-1] = [final_dis_x, mid_y]

        # terrain.height_field_raw[final_dis_x:round(length_x/terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0
        
        return terrain, goals, final_dis_x
    

    def hurdle(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.2],
            hurdle_height_range=[0.05, 0.15],
            flat_size = 0.6
            ):
        """
        🏃‍♂️ 跨栏障碍地形生成器
        
        评估场景对应：Normal Terrain（正常地形）
        
        功能描述：
        创建一系列连续的跨栏障碍，测试机器人的跨越和节奏控制能力。
        机器人需要保持稳定的步态，连续跨越多个障碍物。
        
        训练目标：
        - 跨越动作的协调性
        - 步态节奏的一致性  
        - 连续障碍的适应能力
        - 落地缓冲和平衡控制
        
        难度调节：
        - difficulty=0.0: 低矮障碍，间距较大，易于跨越
        - difficulty=1.0: 高障碍，间距紧密，需要精确控制
        
        与其他地形的区别：
        - parkour(): 需要跳跃到不连续的石头平台
        - hurdle(): 跨越连续的低矮障碍，地面连续
        - gap(): 需要跳跃跨越深坑间隙
        """
        
        # 初始化目标点数组
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  # 地形中线Y坐标
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals  # 每个目标点间的X距离

        # 根据难度计算障碍物参数
        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)  # 障碍宽度
        hurdle_height = round(((hurdle_height_range[1]-hurdle_height_range[0])*difficulty + hurdle_height_range[0])/terrain.vertical_scale)  # 障碍高度

        platform_size = round(platform_size / terrain.horizontal_scale)
        # terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0

        # 创建平坦的基础地形
        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)  # 障碍间的平坦区域大小
        dis_x = start_x + platform_size  # 当前障碍的X位置

        # 设置目标点：均匀分布在跨栏路径上
        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        # 生成连续的跨栏障碍
        for i in range(num_goals):
            # 在当前位置创建跨栏障碍（横跨整个宽度）
            terrain.height_field_raw[dis_x-hurdle_size//2:dis_x+hurdle_size//2, start_y:start_y+mid_y*2] = hurdle_height
            dis_x += flat_size + hurdle_size  # 移动到下一个障碍位置

        return terrain,goals,dis_x
        
  
    def bridge(terrain,
               length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                bridge_width_range=[0.3,0.4],  
                bridge_height=0.7,
                ):
        """
        🌉 窄桥行走地形生成器
        
        评估场景对应：Challenging Terrain（挑战地形）
        
        功能描述：
        创建一个窄桥，两侧是深坑，测试机器人在狭窄空间内的精确行走能力。
        这是最考验机器人平衡控制和路径跟踪精度的地形之一。
        
        训练目标：
        - 高精度的直线行走能力
        - 侧向平衡控制
        - 恐高环境下的心理适应（对AI而言是传感器噪声适应）
        - 窄空间约束下的步态调整
        
        挑战特点：
        - 容错空间极小：一步偏离即可能跌落
        - 需要持续的侧向平衡修正
        - 要求稳定而谨慎的前进步态
        
        难度调节：
        - difficulty=0.0: 桥面较宽（0.4m），相对安全
        - difficulty=1.0: 桥面很窄（0.3m），极具挑战性
        
        在组合地形中的作用：
        - 作为连接不同地形区域的"瓶颈"
        - 测试从开阔地形到约束空间的适应能力
        - 在Generalization Evaluation中作为关键挑战点
        """
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y / terrain.horizontal_scale) // 2  # 地形中线
        
        # 根据难度计算桥宽（难度越高桥越窄）
        bridge_width = round(((bridge_width_range[1]-bridge_width_range[0])*difficulty +bridge_width_range[0])/terrain.horizontal_scale)
        bridge_height = round(bridge_height / terrain.vertical_scale)  # 深坑深度
        platform_size = round(platform_size / terrain.horizontal_scale)
        
        # 创建起始平台
        terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0
        
        bridge_start_x = platform_size + start_x
        bridge_length = round(length_x / terrain.horizontal_scale)
        bridge_end_x = start_x + bridge_length

        # 设置目标点：沿着桥的中心线均匀分布
        for i in range(num_goals):
            goals[i] = [bridge_start_x + bridge_length/num_goals*i, mid_y]  
       
        # 计算桥两侧深坑的边界
        left_y1 = 0
        left_y2 = int(mid_y - bridge_width // 2)   # 左侧深坑右边界
        right_y1 = int(mid_y + bridge_width // 2)  # 右侧深坑左边界
        right_y2 = mid_y*2
        
        # 创建两侧深坑（桥面保持在原始高度0）
        terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y1:left_y2] = -bridge_height    # 左侧深坑
        terrain.height_field_raw[bridge_start_x:bridge_end_x, right_y1:right_y2] = -bridge_height  # 右侧深坑

        # terrain.height_field_raw[bridge_start_x:bridge_end_x, left_y2:right_y1] = 0  # 桥面（已经是0）

        return terrain,goals,bridge_end_x

    
    def flat(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            ):
        """
        🏃 平坦地形生成器
        
        评估场景对应：Simple Terrain（简单地形）
        
        功能描述：
        创建完全平坦的地形，作为基础运动能力的评估基准。
        这是最基本的地形类型，用于验证机器人的基本行走能力。
        
        训练目标：
        - 稳定的直线行走
        - 基本步态模式的建立
        - 速度控制和方向控制
        - 能量效率优化
        
        在评估体系中的作用：
        - 作为性能基准线：其他复杂地形的表现都以此为参考
        - 验证基本功能：确保机器人具备最基本的移动能力
        - 调试工具：在复杂地形失败时，用于排查基础问题
        
        在组合地形中的作用：
        - 作为不同挑战区域间的"休息区"
        - 提供重新调整步态和姿态的机会
        - 在Generalization Evaluation中作为连接段
        """
        goals = np.zeros((num_goals, 2))
        length_x = round(length_x / terrain.horizontal_scale)  # 转换为网格单位
        length_y = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size / terrain.horizontal_scale)

        # 设置目标点：沿直线均匀分布
        for i in range(num_goals):
            # y_pos = round(random.uniform(0,length_y))  # 可选：随机Y位置
            y_pos = length_y//2  # 沿中线设置目标点
            goals[i]=[start_x+platform_size+length_x/num_goals*i,start_y+y_pos]

        # 地形已经是平坦的（高度为0），无需额外处理
        return terrain,goals,length_x


    def uneven(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            num_range=[150,200],
            size_range=[0.4,0.7],
            height_range=[0.1,0.2],
            ):   

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals
        mid_y = round(length_y/ terrain.horizontal_scale) // 2

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+per_x*i,start_y+mid_y]

        height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)


        min_size = round(size_range[0]/ terrain.horizontal_scale)
        max_size = round(size_range[1]/ terrain.horizontal_scale)

        discrete_start_x = start_x+platform_size
        discrete_start_y = start_y

        discrete_end_x = discrete_start_x +round(length_x/ terrain.horizontal_scale) - platform_size
        discrete_end_y = discrete_start_y +round(length_y/ terrain.horizontal_scale)

        num_rects = round((num_range[1]-num_range[0])*difficulty + num_range[0])

        for _ in range(num_rects):
            width = round(random.uniform(min_size, max_size))
            length = round(random.uniform(min_size, max_size))
            start_i = round(random.uniform(discrete_start_x, discrete_end_x-width))
            start_j = round(random.uniform(discrete_start_y, discrete_end_y-length))

            terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = random.uniform(-height//2, height)

        terrain.height_field_raw[start_x:start_x+platform_size , start_y:start_y+mid_y*2] = 0
        terrain.height_field_raw[discrete_end_x:discrete_end_x+platform_size , start_y:start_y+mid_y*2] = 0

        return terrain,goals,discrete_end_x+platform_size

   
    def stair(terrain,
                length_x=18.0,
                length_y=4.0,
                num_goals=8,
                start_x = 0,
                start_y = 0,
                platform_size=1.0, 
                difficulty = 0.5,
                height_range=[0.08,0.2],
                size_range=[0.4,0.5],
                upstair = True,
                start_z = 3.0
                ):
        """
        🪜 阶梯地形生成器 - 鲁棒性与极限评估核心地形
        
        评估场景对应：
        - Robustness Evaluation（鲁棒性评估）：扩展阶梯数量测试持续爬升能力
        - Extreme Evaluation（极限评估）：增加阶梯高度测试运动能力上限
        
        功能描述：
        创建上行或下行阶梯，是评估机器人垂直运动能力的标准地形。
        通过调整阶梯数量和高度，可以进行不同强度的评估。
        
        评估模式详解：
        
        🔄 鲁棒性评估模式：
        - 训练时：5-10个标准阶梯
        - 测试时：扩展到20+个阶梯
        - 目标：测试长距离连续爬升的稳定性和耐久性
        - 关键指标：是否能保持稳定步态，避免累积误差
        
        ⚡ 极限评估模式：
        - 训练时：阶梯高度0.08-0.2m
        - 测试时：显著增加阶梯高度
        - 目标：探索机器人运动能力的物理极限
        - 关键指标：最大可攀爬高度，失败模式分析
        
        训练目标：
        - 垂直方向的重心控制
        - 台阶边缘的精确踏步
        - 上下肢协调的爬升步态
        - 高度变化的视觉感知
        
        难度调节：
        - difficulty=0.0: 低矮阶梯(0.08m)，台阶较大(0.5m)
        - difficulty=1.0: 高阶梯(0.2m)，台阶较小(0.4m)
        
        参数说明：
        - upstair=True: 上行阶梯（爬升挑战）
        - upstair=False: 下行阶梯（下降控制挑战）
        - start_z: 下行时的起始高度
        """

        goals = np.zeros((num_goals, 2))
        platform_size = round(platform_size/ terrain.horizontal_scale)
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals  # 每个目标点的X间距
        per_y = round(length_y/ terrain.horizontal_scale) // 2  # 地形中线Y坐标
        
        # 根据难度计算阶梯参数
        step_height = round(((height_range[1]-height_range[0])*difficulty + height_range[0])/terrain.vertical_scale)  # 单个阶梯高度
        step_x = round(((size_range[0]-size_range[1])*difficulty +size_range[1])/terrain.horizontal_scale)  # 单个阶梯深度

        # 初始化累积高度
        if(upstair):
            total_step_height = 0  # 上行从0开始
        else:
            total_step_height = round(start_z/terrain.vertical_scale)  # 下行从起始高度开始

        dis_x = start_x + platform_size  # 第一个阶梯的起始位置

        # 设置目标点：每个目标点位于对应阶梯的中心
        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+per_y]

        # 生成阶梯序列
        for i in range(num_goals):
            if(upstair):
                total_step_height += step_height  # 上行：逐步增高
            else :
                total_step_height -= step_height  # 下行：逐步降低

            # 创建当前阶梯（横跨整个宽度）
            terrain.height_field_raw[dis_x : dis_x + step_x, start_y : start_y + per_y*2] = total_step_height
            dis_x += step_x  # 移动到下一个阶梯位置

        # terrain.height_field_raw[start_x:start_x+platform_size,start_y:start_y + per_y*2] = 0  # 起始平台
        
        # 创建终点平台（保持最终高度）
        terrain.height_field_raw[dis_x:start_x+round(length_x/ terrain.horizontal_scale),start_y:start_y + per_y*2] = total_step_height

        return terrain,goals,start_x+round(length_x/ terrain.horizontal_scale)


    def wave(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            amplitude_range=[0.05,0.1]
            ):   
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2
        platform_size = round(1.5/ terrain.horizontal_scale)
        mid_x =  (round(length_x/ terrain.horizontal_scale) - platform_size)// num_goals

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]
        
        x_indices = np.arange(start_x, start_x + mid_x*num_goals + platform_size)
        amplitude = round(((amplitude_range[1]-amplitude_range[0])*difficulty + amplitude_range[0])/terrain.vertical_scale)
        wave_pattern = amplitude * np.sin(2 * np.pi * x_indices / length_x)

        for i, wave_height in enumerate(wave_pattern):
            terrain.height_field_raw[x_indices[i], start_y:start_y +mid_y*2] = wave_height

        terrain.height_field_raw[start_x :start_x + platform_size, start_y:start_y+ mid_y*2] = 0

        return terrain,goals,start_x+mid_x*num_goals

    
    def slope(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0, 
            difficulty = 0.5,
            angle_range = [4.1,10.0],
            uphill=False
            ):    

        goals = np.zeros((num_goals, 2))
        length_x_grid = round((length_x - platform_size) / terrain.horizontal_scale)
        length_y_grid = round(length_y / terrain.horizontal_scale)
        platform_size = round(platform_size/ terrain.horizontal_scale)

        for i in range(num_goals):
            goals[i]=[start_x+platform_size+length_x_grid/num_goals*i,start_y+length_y_grid//2]

        slope_angle = (angle_range[1]-angle_range[0])*difficulty + angle_range[0]
        angle_rad = math.radians(slope_angle)
        total_height = length_x * math.tan(angle_rad)

        total_height_units = total_height / terrain.vertical_scale

        start_x += platform_size

        for x in range(start_x, start_x + length_x_grid):
            progress = (x - start_x) / length_x_grid
            if uphill:
                height = progress * total_height_units
            else:
                height = (1 - progress) * total_height_units
            terrain.height_field_raw[x, start_y:start_y + length_y_grid] = round(height)
        
        return terrain,goals,start_x + length_x_grid

 
    def gap(terrain,
            length_x=18.0,
            length_y=4.0,
            num_goals=8,
            start_x = 0,
            start_y = 0,
            platform_size=1.0,
            difficulty = 0.5,
            gap_height = 2.,
            gap_low_range = [0.15,0.3],
            ):
        """
        🕳️ 间隙跳跃地形生成器
        
        评估场景对应：Hard Terrain（困难地形）
        
        功能描述：
        创建一系列深坑间隙，机器人需要进行跳跃才能通过。
        测试机器人的爆发力、跳跃距离控制和着陆稳定性。
        
        训练目标：
        - 跳跃起跳的力量控制
        - 空中姿态的调整能力
        - 着陆时的冲击缓冲
        - 跳跃距离的精确估算
        
        与其他地形的区别：
        - parkour(): 跳到石头平台上，有明确的着陆目标
        - gap(): 跳跃跨越深坑，着陆在平地上
        - hurdle(): 跨越低矮障碍，脚不离地太久
        
        挑战特点：
        - 需要瞬间爆发力
        - 空中时间较长，需要姿态控制
        - 着陆精度要求高
        - 连续跳跃的节奏掌握
        
        难度调节：
        - difficulty=0.0: 间隙较小(0.3m)，容易跳跃
        - difficulty=1.0: 间隙较大(0.15m)，需要更强跳跃能力
        
        注意：gap_low_range的逻辑是反向的，difficulty越高间隙越小
        """
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale) //2  # 地形中线
        mid_x =  round((length_x - platform_size)/ terrain.horizontal_scale) // num_goals  # 每个目标点间距
        platform_size = round(platform_size/ terrain.horizontal_scale)

        # 设置目标点：沿中线均匀分布
        for i in range(num_goals):
            goals[i]=[start_x+platform_size+mid_x*i,start_y+mid_y]

        # 根据难度计算间隙大小（注意：这里是反向逻辑）
        gap_size = round(( (gap_low_range[0]-gap_low_range[1])*difficulty + gap_low_range[1] )/terrain.horizontal_scale)
        gap_dis_x = start_x + platform_size + gap_size  # 第一个间隙的位置
        gap_dis_y = start_y + mid_y  # 间隙的Y中心位置
        
        # 创建一系列间隙（深坑）
        for i in range(num_goals):
            # 在当前位置挖一个深坑
            terrain.height_field_raw[gap_dis_x :gap_dis_x + gap_size, gap_dis_y - mid_y:gap_dis_y + mid_y] = -round(gap_height / terrain.vertical_scale)
            gap_dis_x += 3*gap_size  # 移动到下一个间隙（间隔为3倍gap_size）
        
        # 确保起始平台是平坦的
        terrain.height_field_raw[start_x :start_x + platform_size, start_y :start_y + mid_y*2] = 0

        return terrain, goals,start_x+mid_x*num_goals
    
 
    def plot(
            terrain,
            length_x=18.,
            length_y=4.,
            num_goals=8,
            start_x=0,
            start_y=0,
            platform_size=1., 
            difficulty = 0.5,
            hurdle_range=[0.1, 0.15],
            hurdle_height = 1.2,
            flat_size = 1.0
            ):
        
        goals = np.zeros((num_goals, 2))
        mid_y = round(length_y/ terrain.horizontal_scale)// 2  
        per_x = (round(length_x/ terrain.horizontal_scale)- platform_size) // num_goals


        hurdle_size = round(((hurdle_range[1]-hurdle_range[0])*difficulty +hurdle_range[0])/terrain.horizontal_scale)// 2
        hurdle_height = round(hurdle_height/terrain.vertical_scale)

        platform_size = round(platform_size / terrain.horizontal_scale)
        # terrain.height_field_raw[start_x:start_x+platform_size, start_y:start_y+2*mid_y] = 0

        terrain.height_field_raw[start_x:start_x +round(length_x/ terrain.horizontal_scale), start_y:start_y+mid_y*2] = 0

        flat_size = round(flat_size / terrain.horizontal_scale)
        dis_x = start_x + platform_size

        for i in range(num_goals):
            goals[i]=[dis_x+per_x*i,start_y+mid_y]

        for i in range(num_goals):

            terrain.height_field_raw[dis_x-hurdle_size:dis_x+hurdle_size, start_y+mid_y - hurdle_size:start_y+mid_y + hurdle_size] = hurdle_height
            dis_x += flat_size + hurdle_size * 2

        return terrain,goals,dis_x
