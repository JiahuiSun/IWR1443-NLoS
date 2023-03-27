import numpy as np


# 我要做的几件事：读数据、NLoS过滤和映射、可视化
def line_by_2p(p1, p2):
    """A ray from p1 to p2.
    """
    ABC = np.array([p2[1]-p1[1], p1[0]-p2[0], p2[0]*p1[1]-p1[0]*p2[1]])
    if p1[1] > p2[1]:
        ABC = -ABC
    return ABC


def nlosFilterAndMapping(pointCloud, radar_pos, corner_args):
    point_cloud_ext = np.concatenate([pointCloud[:, :2], np.ones((pointCloud.shape[0], 1))], axis=1)
    top_wall_y = corner_args['top_wall_y']
    bottom_wall_y = corner_args['bottom_wall_y']
    left_wall_x = corner_args['left_wall_x']

    top_map_bottom_y = 2*top_wall_y - bottom_wall_y
    top_map_radar = [radar_pos[0], 2*top_wall_y-radar_pos[1]]
    left_border = line_by_2p(radar_pos, [left_wall_x, bottom_wall_y])
    right_border = line_by_2p(top_map_radar, [left_wall_x, top_map_bottom_y])
    top_border = top_map_bottom_y
    flag1 = point_cloud_ext.dot(left_border) > 0
    flag2 = point_cloud_ext.dot(right_border) < 0
    flag3 = pointCloud[:, 1] < top_border
    flag = np.logical_and(np.logical_and(flag1, flag2), flag3)
    pointCloud[flag, 1] = 2*top_wall_y - pointCloud[flag, 1]
    point_cloud_filter = pointCloud[flag, :]
    return point_cloud_filter


def transform(radar_xy, delta_x, delta_y, yaw):
    """Transform xy from radar coordinate to the world coordinate.
    Inputs:
        radar_xy: Nx2
    Return:
        world_xy: Nx2
    """
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([delta_x, delta_y]).reshape(-1, 1)
    world_xy = (rotation_matrix.dot(radar_xy.T) + translation_vector).T
    return world_xy
