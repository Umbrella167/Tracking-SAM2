import numpy as np
import cv2
from scipy.spatial import KDTree

def draw_uniform_points(frame, mask, point_num):
    """
    在掩码区域内均匀绘制指定数量的点。

    Args:
        frame: 原始图像。
        mask: 掩码图像，非零区域表示有效区域。
        point_num: 要绘制的点的数量。

    Returns:
        绘制了点的图像副本，以及所选点的坐标列表。
    """
    points = np.argwhere(mask != 0)
    if len(points) == 0 or point_num <= 0:
        return frame.copy(), []

    coords = points[:, [1, 0]].astype(np.float32)  # 转换为(x,y)格式
    selected_coords = poisson_disk_sampling(coords, point_num)
    
    img = frame.copy()
    for (x, y) in selected_coords:
        cv2.circle(img, (int(x), int(y)), radius=3, color=(0, 0, 255), thickness=-1)
    return img, selected_coords

def poisson_disk_sampling(coords, num_samples, radius=15):
    """
    使用KD-Tree加速的泊松盘采样算法。

    Args:
        coords: 候选点坐标数组，形状为(N, 2)。
        num_samples: 需要采样的点数。
        radius: 最小间距。

    Returns:
        采样后的点坐标列表。
    """
    if num_samples >= len(coords):
        return coords.tolist()
    
    coords = np.asarray(coords)
    if len(coords) == 0:
        return []
    
    # 随机选择初始点
    selected_indices = [np.random.choice(len(coords))]
    mask = np.ones(len(coords), dtype=bool)
    mask[selected_indices[0]] = False

    # 构建初始KD-Tree
    selected_points = [coords[selected_indices[0]]]
    tree = KDTree(selected_points)

    while len(selected_points) < num_samples:
        # 获取当前可用点
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            break

        # 批量查询所有可用点的最近距离
        distances, _ = tree.query(coords[valid_indices], k=1)
        valid_mask = (distances >= radius).flatten()

        # 筛选合格候选点
        candidates = valid_indices[valid_mask]
        if len(candidates) == 0:
            # 尝试逐步减小半径
            radius *= 0.9
            continue

        # 找到最大最小距离的候选点
        best_candidate = candidates[np.argmax(distances[valid_mask])]
        
        # 更新数据
        mask[best_candidate] = False
        selected_points.append(coords[best_candidate])
        selected_indices.append(best_candidate)
        
        # 增量更新KD-Tree（重新构建）
        tree = KDTree(selected_points)

    return selected_points[:num_samples]