import serial
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt


# Constants
maxBufferSize = 2**15
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0
MMWDEMO_UART_MSG_DETECTED_POINTS = 1
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]


# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName, CLIPort, dataPort):
    CLIPort = serial.Serial(CLIPort, 115200)
    dataPort = serial.Serial(dataPort, 921600)
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIPort.write((i+'\n').encode())
        print(i)
        time.sleep(0.01)  # 发送命令时间隔一段时间，防止内存溢出
    return CLIPort, dataPort


# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}
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
    return configParameters
   

# Funtion to read and parse the incoming data
def readAndParseData14xx(fileName, configParameters):    
    global byteBufferLength, byteBuffer
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}
    rawDataFile = open(fileName, "rb")
    readBuffer = rawDataFile.read()
    rawDataFile.close()
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
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
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
                # Store the data in the detObj dictionary
                detObj = {"numObj": tlv_numObj, "rangeIdx": rangeIdx, "range": rangeVal, "dopplerIdx": dopplerIdx, \
                          "doppler": dopplerVal, "peakVal": peakVal, "x": x, "y": y, "z": z}
                dataOK = 1 
        
        # Remove already processed data
        if idX > 0 and byteBufferLength > idX:
            shiftSize = totalPacketLen
            byteBuffer[:byteBufferLength-shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength-shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength-shiftSize:]), dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
    print(f"byteBufferLength={byteBufferLength}")
    return dataOK, frameNumber, detObj


def nlosFilterAndMapping():
    pass


def transform():
    pass


def visualize():
    pass


def main(args):
    configParameters = parseConfigFile(args.config)
    currentIndex = 0
    fig, ax = plt.subplots()
    plt.ion()
    while True:
        dataOk, frameNumber, detObj = readAndParseData14xx(args.fileName, configParameters)
        if dataOk:
            currentIndex += 1
            if len(detObj["x"]) > 0:
                x = -detObj["x"]
                y = detObj["y"]
                ax.scatter(x, y)
                ax.set_title(f"frameNumber={frameNumber}, currIdx={currentIndex}")
                ax.set_xlabel("x(m)")
                ax.set_ylabel("y(m)")
                plt.pause(1e-2)
                ax.cla()
        # TODO: 为啥要sleep啊，难道不能读取太快？这里指的是什么采样频率是30Hz啊？
        time.sleep(0.033) # Sampling frequency of 30 Hz


if __name__ == "__main__":
    parser = argparse.ArgumentParser("mmWave NLoS sensing")
    parser.add_argument("--config", type=str, default="profiles/profile.cfg")
    parser.add_argument("--fileName", type=str, default="")
    args = parser.parse_args()
    main(args)
