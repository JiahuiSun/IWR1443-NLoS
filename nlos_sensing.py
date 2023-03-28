import numpy as np


def line_by_2p(p1, p2):
    """A ray from p1 to p2.
    """
    ABC = np.array([p2[1]-p1[1], p1[0]-p2[0], p2[0]*p1[1]-p1[0]*p2[1]])
    if p1[1] > p2[1]:
        ABC = -ABC
    return ABC


def nlosFilterAndMapping(pointCloud, radar_pos, corner_args):
    """TODO: 现在只实现了L型转角
    """
    point_cloud_ext = np.concatenate([pointCloud[:, :2], np.ones((pointCloud.shape[0], 1))], axis=1)
    top_wall_y = corner_args['top_wall_y']
    side_wall_x = corner_args['side_wall_x']
    inner_corner = corner_args['inner_corner']
    # Filter
    top_map_corner = [inner_corner[0], 2*top_wall_y - inner_corner[1]]
    top_map_radar = [radar_pos[0], 2*top_wall_y-radar_pos[1]]
    # 用前墙反射
    top_border = top_map_corner[1]
    # 如果转角在雷达左边
    if inner_corner[0] < radar_pos[0]:
        left_border = line_by_2p(radar_pos, inner_corner)
        right_border = line_by_2p(top_map_radar, top_map_corner)
    # 如果转角在雷达右边
    elif inner_corner[0] > radar_pos[0]:
        left_border = line_by_2p(top_map_radar, top_map_corner)
        right_border = line_by_2p(radar_pos, inner_corner)
    flag1 = point_cloud_ext.dot(left_border) > 0
    flag2 = point_cloud_ext.dot(right_border) < 0
    flag3 = pointCloud[:, 1] < top_border
    flag = np.logical_and(np.logical_and(flag1, flag2), flag3)
    # Mapping
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
