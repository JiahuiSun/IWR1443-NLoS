import serial
import time
import numpy as np
import argparse
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from multiprocessing import Queue

from read_IWR1443 import read_IWR1443


# Global variables
data_queue = Queue()
CLIPort = 0
dataPort = 0
configParameters = {}
# radar related
person_pos_key = "1_0"
radar_pos_key = "1-0"
radar_angle = 45
corner_args = {
    'top_wall_y': 5.32, 
    'bottom_wall_y': 3.18, 
    'left_wall_x': -1.8
}
if person_pos_key == "1_0":
    ground_truth = [-3.6, 4.8]
elif person_pos_key == "1_1":
    ground_truth = [-4.8, 4.8]
elif person_pos_key == "1_2":
    ground_truth = [-6.0, 4.8]
elif person_pos_key == "2_0":
    ground_truth = [-3.6, 3.6]
elif person_pos_key == "2_1":
    ground_truth = [-4.8, 3.6]
elif person_pos_key == "2_2":
    ground_truth = [-6.0, 3.6]
else:
    raise Exception("Wrong person position")

if radar_pos_key == "1-0":
    radar_pos = [-0.6, 2.4]
elif radar_pos_key == "1-1":
    radar_pos = [-0.6, 1.2]
elif radar_pos_key == "1-2":
    radar_pos = [-0.6, 0.0]
elif radar_pos_key == "2-0":
    radar_pos = [0.0, 2.4]
elif radar_pos_key == "2-1":
    radar_pos = [0.0, 1.2]
elif radar_pos_key == "2-2":
    radar_pos = [0.0, 0.0]
else:
    raise Exception("Wrong radar position")

top_wall_y = 5.32
bottom_wall_y = 3.18
left_wall_x = -1.8
fov_line_k = (bottom_wall_y - radar_pos[1]) / (left_wall_x - radar_pos[0])
fov_line_z = radar_pos[1] - fov_line_k * radar_pos[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
line0, = ax[0].plot([], [], 'ob', ms=5)
line1, = ax[0].plot([], [], 'or', ms=5)
line2, = ax[1].plot([], [], 'ob', ms=5)
line3, = ax[1].plot([], [], 'or', ms=5)
lines = [line0, line1, line2, line3]


# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName, cPort, dPort):
    global CLIPort, dataPort
    CLIPort = serial.Serial(cPort, 115200)
    dataPort = serial.Serial(dPort, 921600)
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIPort.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)  # 发送命令时间隔一段时间，防止内存溢出


# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    global configParameters
    configParameters['numTxAnt'] = 3
    configParameters['numRxAnt'] = 4
    numTxAnt = 3
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        splitWords = i.split(" ")
        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
            digOutSampleRate = int(splitWords[11])
        # Get the information about the frame configuration    
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(splitWords[5])
    # Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)


def line_by_2p(p1, p2):
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
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([delta_x, delta_y]).reshape(-1, 1)
    world_xy = (rotation_matrix.dot(radar_xy.T) + translation_vector).T
    return world_xy


def init():
    for i in range(2):
        ax[i].set_xlabel('x(m)')
        ax[i].set_ylabel('y(m)')
        ax[i].plot(ground_truth[0], ground_truth[1], '*g', ms=10)
        ax[i].plot(radar_pos[0], radar_pos[1], 'dc', ms=10)
        ax[i].set_xlim([-5, 5])
        ax[i].set_ylim([0, 10])
        ax[i].plot([-5, 5], [top_wall_y, top_wall_y], 'k')
        ax[i].plot([-5, left_wall_x], [bottom_wall_y, bottom_wall_y], 'k')
        ax[i].plot([left_wall_x, left_wall_x], [bottom_wall_y, 0], 'k')
        ax[i].plot([radar_pos[0], (top_wall_y - fov_line_z) / fov_line_k], [radar_pos[1], top_wall_y], 'k')
    return lines


def visualize(data):
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
            point_cloud[:, :2] = transform(point_cloud[:, :2], radar_pos[0], radar_pos[1], 360-radar_angle)
            # 过滤并映射
            point_cloud_nlos = nlosFilterAndMapping(point_cloud, radar_pos, corner_args)
            yield frame_num, point_cloud, point_cloud_nlos


def main(args):
    serialConfig(args.config, args.cPort, args.dPort)
    parseConfigFile(args.config)
    try:
        receive_thread = read_IWR1443(data_queue, dataPort, args.read_delay)
        receive_thread.start()
        ani = animation.FuncAnimation(
            fig, visualize, nlosProcess, interval=33,
            init_func=init, repeat=False
        )
        plt.show()
        receive_thread.terminate()
    except KeyboardInterrupt:
        CLIPort.write(('sensorStop\n').encode())
        CLIPort.close()
        dataPort.close()
        print('sensorStop')
        receive_thread.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mmWave NLoS sensing")
    parser.add_argument("--config", type=str, default="profiles/profile.cfg")
    parser.add_argument("--cPort", type=str, default="/dev/ttyACM0")
    parser.add_argument("--dPort", type=str, default="/dev/ttyACM1")
    parser.add_argument("--read_delay", type=float, default=0.1, help="Delay for reading sensor data.")
    args = parser.parse_args()
    main(args)
