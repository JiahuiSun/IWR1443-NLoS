import argparse
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Queue
import numpy as np
import os

from read_IWR1443 import read_IWR1443
from utils import serialConfig, parseConfigFile
from nlos_sensing import transform, nlosFilterAndMapping


# Global variables
data_queue = Queue()
configParameters = {}
save = 30
cnt = 0
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# environment related
radar_pos = [-0.6, 2.4]
radar_angle = 360 - 45  # radar逆时针旋转多少度和世界坐标系重合
ground_truth = [-3.6, 4.8]
inner_corner = [-1.8, 3.18]
top_wall_y = 5.32
side_wall_x = None
corner_args = {
    'top_wall_y': top_wall_y, 
    'side_wall_x': side_wall_x,
    'inner_corner': inner_corner
}
fov_line_k = (inner_corner[1] - radar_pos[1]) / (inner_corner[0] - radar_pos[0])
fov_line_z = radar_pos[1] - fov_line_k * radar_pos[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
line0, = ax[0].plot([], [], 'ob', ms=5)
line1, = ax[0].plot([], [], 'or', ms=5)
line2, = ax[1].plot([], [], 'ob', ms=5)
line3, = ax[1].plot([], [], 'or', ms=5)
lines = [line0, line1, line2, line3]


def init_fig():
    for i in range(2):
        ax[i].set_xlabel('x(m)')
        ax[i].set_ylabel('y(m)')
        ax[i].plot(ground_truth[0], ground_truth[1], '*g', ms=10)
        ax[i].plot(radar_pos[0], radar_pos[1], 'dc', ms=10)
        ax[i].set_xlim([-5, 5])
        ax[i].set_ylim([0, 10])
        ax[i].plot([-5, 5], [top_wall_y, top_wall_y], 'k')
        ax[i].plot([-5, inner_corner[0]], [inner_corner[1], inner_corner[1]], 'k')
        ax[i].plot([inner_corner[0], inner_corner[0]], [inner_corner[1], 0], 'k')
        ax[i].plot([radar_pos[0], (top_wall_y - fov_line_z) / fov_line_k], [radar_pos[1], top_wall_y], 'k')
    return lines


def visualize(data):
    global cnt
    frameNumber, pointCloud, pointCloudNLOS = data
    if pointCloud is not None:
        ax[0].set_title(f"Original Point Cloud, frame={frameNumber}")
        ax[1].set_title(f"NLoS Point Cloud, frame={frameNumber}")
        static_idx = pointCloud[:, 3] == 0
        dynamic_idx = pointCloud[:, 3] != 0
        lines[0].set_data(pointCloud[static_idx, 0], pointCloud[static_idx, 1])
        lines[1].set_data(pointCloud[dynamic_idx, 0], pointCloud[dynamic_idx, 1])
        static_idx = pointCloudNLOS[:, 3] == 0
        dynamic_idx = pointCloudNLOS[:, 3] != 0
        lines[2].set_data(pointCloudNLOS[static_idx, 0], pointCloudNLOS[static_idx, 1])
        lines[3].set_data(pointCloudNLOS[dynamic_idx, 0], pointCloudNLOS[dynamic_idx, 1])
        # 保存save帧有点云的帧
        if save > 0 and cnt < save:
            np.save(os.path.join(output_dir, f'frame{frameNumber}.npy'), pointCloudNLOS)
        cnt += 1
    return lines


def nlosProcess():
    while True:
        print(f"队列长度: {data_queue.qsize()}")
        if data_queue.empty():
            yield None, None, None
            continue
        results = data_queue.get()
        frame_num, point_cloud = results['frame_num'], results.get('detected_objects')
        if point_cloud is None:
            yield None, None, None
        else:
            # 计算速度
            dopplerIdx = point_cloud[:, -1]
            dopplerIdx[dopplerIdx>(configParameters["numDopplerBins"]/2-1)] = dopplerIdx[dopplerIdx>(configParameters["numDopplerBins"]/2-1)] - 65535
            point_cloud[:, -1] = dopplerIdx * configParameters["dopplerResolutionMps"]
            # point_cloud_nlos = point_cloud
            # 坐标变换
            point_cloud[:, :2] = transform(point_cloud[:, :2], radar_pos[0], radar_pos[1], radar_angle)
            # 过滤并映射
            point_cloud_nlos = nlosFilterAndMapping(point_cloud, radar_pos, corner_args)
            yield frame_num, point_cloud, point_cloud_nlos


def main(args):
    global configParameters
    CLIPort, dataPort = serialConfig(args.config, args.cPort, args.dPort)
    configParameters = parseConfigFile(args.config)
    try:
        receive_thread = read_IWR1443(data_queue, dataPort, args.read_interval)
        receive_thread.start()
        ani = animation.FuncAnimation(
            fig, visualize, nlosProcess, interval=args.proc_interval,
            init_func=init_fig, repeat=False, save_count=100
        )
        plt.show()
        ani.save(os.path.join(output_dir, "ani.gif"), writer='imagemagick')
        receive_thread.terminate()
        CLIPort.write(('sensorStop\n').encode())
        CLIPort.close()
        dataPort.close()
    except KeyboardInterrupt:
        receive_thread.terminate()
        CLIPort.write(('sensorStop\n').encode())
        CLIPort.close()
        dataPort.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mmWave NLoS sensing")
    parser.add_argument("--config", type=str, default="profiles/profile.cfg")
    parser.add_argument("--cPort", type=str, default="/dev/ttyACM0")
    parser.add_argument("--dPort", type=str, default="/dev/ttyACM1")
    parser.add_argument("--read_interval", type=float, default=0.1, help="Delay for reading sensor data (s).")
    parser.add_argument("--proc_interval", type=float, default=50, help="Delay for reading sensor data (ms).")
    args = parser.parse_args()
    main(args)
