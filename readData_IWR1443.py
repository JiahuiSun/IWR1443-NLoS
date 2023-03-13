import serial
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Global variables
maxBufferSize = 2**15
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0
MMWDEMO_UART_MSG_DETECTED_POINTS = 1
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
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
fig, ax = plt.subplots(1, 2)
line0, = ax[0].plot([], [], 'ob', ms=5)
line1, = ax[1].plot([], [], 'ob', ms=5)
line = [line0, line1]


def init():
    for i in range(2):
        ax[i].set_xlabel('x(m)')
        ax[i].set_ylabel('y(m)')
        ax[i].plot(ground_truth[0], ground_truth[1], '*g', ms=10)
        ax[i].plot(radar_pos[0], radar_pos[1], 'dc', ms=10)
        ax[i].set_xlim([-5, 5])
        ax[i].set_ylim([0, 10])
        ax[i].plot([-5, 5], [top_wall_y, top_wall_y])
        ax[i].plot([-5, left_wall_x], [bottom_wall_y, bottom_wall_y])
        ax[i].plot([left_wall_x, left_wall_x], [bottom_wall_y, 0])
        ax[i].plot([radar_pos[0], (top_wall_y - fov_line_z) / fov_line_k], [radar_pos[1], top_wall_y])
    ax[0].set_title("Original Point Cloud")
    ax[1].set_title("NLoS Point Cloud")


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
   

# Funtion to read and parse the incoming data
def readAndParseData14xx():    
    global byteBufferLength, byteBuffer
    while True:
        magicOK = 0 # Checks if magic number has been read
        frameNumber = 0
        pointCloud, pointCloudNLOS = None, None

        readBuffer = dataPort.read(dataPort.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)
        # Check that the buffer is not full, and then add the data to the buffer
        if (byteBufferLength + byteCount) < maxBufferSize:
            byteBuffer[byteBufferLength:byteBufferLength+byteCount] = byteVec
            byteBufferLength = byteBufferLength + byteCount
        # Check that the buffer has some data
        if byteBufferLength > 16:
            # Check for all possible locations of the magic word
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]
            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc+8]
                if np.all(check == magicWord):
                    startIdx.append(loc)
            # Check that startIdx is not empty
            if startIdx:
                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                    byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]), dtype='uint8')
                    byteBufferLength = byteBufferLength - startIdx[0]
                # Check that there have no errors with the byte buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0
                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]
                # Read the total packet length
                totalPacketLen = np.matmul(byteBuffer[12:12+4], word)
                # Check that all the packet has been read
                if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                    magicOK = 1
        
        # If magicOK is equal to 1 then process the message
        if magicOK:
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            # Initialize the pointer index
            idX = 0
            # Read the header
            magicNumber = byteBuffer[idX:idX+8]
            idX += 8
            version = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            totalPacketLen = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            platform = format(np.matmul(byteBuffer[idX:idX+4], word), 'x')
            idX += 4
            frameNumber = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            timeCpuCycles = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            numDetectedObj = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            numTLVs = np.matmul(byteBuffer[idX:idX+4], word)
            idX += 4
            # subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
            # idX += 4
            # Read the TLV messages
            for tlvIdx in range(numTLVs):
                # word array to convert 4 bytes to a 32 bit number
                word = [1, 2**8, 2**16, 2**24]
                # Check the header of the TLV message
                tlv_type = np.matmul(byteBuffer[idX:idX+4], word)
                idX += 4
                tlv_length = np.matmul(byteBuffer[idX:idX+4], word)
                idX += 4
                # Read the data depending on the TLV message
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    # word array to convert 4 bytes to a 16 bit number
                    word = [1, 2**8]
                    tlv_numObj = np.matmul(byteBuffer[idX:idX+2], word)
                    idX += 2
                    tlv_xyzQFormat = 2**np.matmul(byteBuffer[idX:idX+2], word)
                    idX += 2
                    # Initialize the arrays
                    rangeIdx = np.zeros(tlv_numObj, dtype='int16')
                    dopplerIdx = np.zeros(tlv_numObj, dtype='int16')
                    peakVal = np.zeros(tlv_numObj, dtype='int16')
                    x = np.zeros(tlv_numObj, dtype='int16')
                    y = np.zeros(tlv_numObj, dtype='int16')
                    z = np.zeros(tlv_numObj, dtype='int16')
                    for objectNum in range(tlv_numObj):
                        rangeIdx[objectNum] =  np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        dopplerIdx[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        peakVal[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        x[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        y[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                        z[objectNum] = np.matmul(byteBuffer[idX:idX+2], word)
                        idX += 2
                    # Make the necessary corrections and calculate the rest of the data
                    rangeVal = rangeIdx * configParameters["rangeIdxToMeters"]
                    # TODO: 为啥？
                    dopplerIdx[dopplerIdx>(configParameters["numDopplerBins"]/2-1)] = dopplerIdx[dopplerIdx>(configParameters["numDopplerBins"]/2-1)] - 65535
                    dopplerVal = dopplerIdx * configParameters["dopplerResolutionMps"]
                    x = x / tlv_xyzQFormat
                    y = y / tlv_xyzQFormat
                    z = z / tlv_xyzQFormat
                    # Store the data in the pointCloud dictionary
                    pointCloud = np.array([x, y, z, dopplerVal]).T
                    pointCloudNLOS = pointCloud
                    # # 坐标变换
                    # pointCloud[:, :2] = transform(pointCloud[:, :2], radar_pos[0], radar_pos[1], 360-radar_angle)
                    # # 过滤并映射
                    # pointCloudNLOS = nlosFilterAndMapping(pointCloud, radar_pos, corner_args)
            # Remove already processed data
            if idX > 0 and byteBufferLength > idX:
                shiftSize = totalPacketLen
                byteBuffer[:byteBufferLength-shiftSize] = byteBuffer[shiftSize:byteBufferLength]
                byteBuffer[byteBufferLength-shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength-shiftSize:]), dtype='uint8')
                byteBufferLength = byteBufferLength - shiftSize
                # Check that there are no errors with the buffer length
                if byteBufferLength < 0:
                    byteBufferLength = 0
        yield frameNumber, pointCloud, pointCloudNLOS


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
    flag = (flag1 and flag2) or flag3
    pointCloud[flag, 1] = 2*top_wall_y - pointCloud[flag, 1]
    point_cloud_filter = pointCloud[flag, :]
    return point_cloud_filter


def transform(radar_xy, delta_x, delta_y, yaw):
    yaw = yaw * np.pi / 180
    rotation_matrix = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    translation_vector = np.array([delta_x, delta_y]).reshape(-1, 1)
    world_xy = (rotation_matrix.dot(radar_xy.T) + translation_vector).T
    return world_xy


def visualize(data):
    frameNumber, pointCloud, pointCloudNLOS = data
    if pointCloud is not None:
        # ax[0].scatter(pointCloud[:, 0], pointCloud[:, 1])
        # ax[1].scatter(pointCloudNLOS[:, 0], pointCloudNLOS[:, 1])
        line[0].set_data(pointCloud[:, 0], pointCloud[:, 1])
        line[1].set_data(pointCloudNLOS[:, 0], pointCloudNLOS[:, 1])
    return line


def main(args):
    serialConfig(args.config, args.cPort, args.dPort)
    parseConfigFile(args.config)
    try:
        ani = animation.FuncAnimation(
            fig, visualize, readAndParseData14xx, interval=33,
            init_func=init, repeat=False
        )
        plt.show()
        # for frameNumber, pointCloud in readAndParseData14xx():
        #     if pointCloud is not None:
        #         x = pointCloud[:, 0]
        #         y = pointCloud[:, 1]
        #         z = pointCloud[:, 2]
        #         vel = pointCloud[:, 3]
        #         visualize(pointCloud, frameNumber)
                # ax.scatter(x, y, z)
                # ax.set_xlim([-5, 5])
                # ax.set_ylim([0, 10])
                # ax.set_zlim([-1, 1])
                # ax.set_title(f"frameNumber={frameNumber}")
                # ax.set_xlabel("x(m)")
                # ax.set_ylabel("y(m)")
                # ax.set_zlabel("z(m)")
                # plt.pause(1e-2)
                # ax.cla()
            # TODO: 为啥要sleep啊，难道不能读取太快？这里指的是什么采样频率是30Hz啊？
            # time.sleep(0.033) # Sampling frequency of 30 Hz
    except KeyboardInterrupt:
        CLIPort.write(('sensorStop\n').encode())
        CLIPort.close()
        dataPort.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mmWave NLoS sensing")
    parser.add_argument("--config", type=str, default="profiles/profile.cfg")
    parser.add_argument("--cPort", type=str, default="/dev/ttyACM0")
    parser.add_argument("--dPort", type=str, default="/dev/ttyACM1")
    args = parser.parse_args()
    main(args)
