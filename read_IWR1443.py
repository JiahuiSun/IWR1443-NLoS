from threading import Thread
from multiprocessing import Process
import struct
import numpy as np
import time


class read_IWR1443(Thread):
    def __init__(self, data_queue, data_port):
        super().__init__()
        self.data_queue = data_queue
        self.data_port = data_port
        self.magic = b'\x02\x01\x04\x03\x06\x05\x08\x07'
        self.header_length = 36

    def find_all_magic(self, data):
        offset_list = []
        i = 0
        while i < len(data):
            if data[i] != self.magic[0]:
                i += 1
            elif data[i:i+8] != self.magic:
                i += 1
            else:
                offset_list.append(i)
                i += 8
        return offset_list

    def parseDetectedObjects(self, data, tlvLength):
        numDetectedObj, xyzQFormat = struct.unpack('2H', data[:4])
        x_vec = np.zeros(numDetectedObj, dtype='int16')
        y_vec = np.zeros(numDetectedObj, dtype='int16')
        z_vec = np.zeros(numDetectedObj, dtype='int16')
        doppler_vec = np.zeros(numDetectedObj, dtype='int16')
        for i in range(numDetectedObj):
            rangeIdx, dopplerIdx, peakVal, x, y, z = struct.unpack('3H3h', data[4+12*i:4+12*i+12])
            x_vec[i] = x*1.0/(1 << xyzQFormat)
            y_vec[i] = y*1.0/(1 << xyzQFormat)
            z_vec[i] = z*1.0/(1 << xyzQFormat)
            doppler_vec[i] = dopplerIdx
        point_cloud = np.array([x_vec, y_vec, z_vec, doppler_vec]).T
        return point_cloud

    def parseRangeProfile(data, tlvLength):
        range_profiles = np.zeros(256)
        for i in range(256):
            rangeProfile = struct.unpack('H', data[2*i:2*i+2])
            range_profiles[i] = rangeProfile[0] * 1.0 * 6 / 8 / (1 << 8)
        return range_profiles

    def parseStats(data, tlvLength):
        interProcess, transmitOut, frameMargin, chirpMargin, activeCPULoad, interCPULoad = struct.unpack('6I', data[:24])
        stats = {
            "ChirpMargin": chirpMargin,
            "FrameMargin": frameMargin,
            "InterCPULoad": interCPULoad,
            "ActiveCPULoad": activeCPULoad,
            "TransmitOut": transmitOut,
            "Interprocess": interProcess
        }
        return stats

    def parse_packet(self, packet):
        # 解析Header
        try:
            magic, version, total_packet_len, platform, frameNum, cpuCycles, numObj, numTLVs = struct.unpack('Q7I', packet[:self.header_length])
        except:
            print("Improper Header structure found")
            return None
        # 判断长度是否足够
        if len(packet) < total_packet_len:
            print("Improper packet length found")
            return None
        # 解析TLV
        packet = packet[self.header_length:]
        results = {'frame_num': frameNum}
        for i in range(numTLVs):
            tlvType, tlvLength = struct.unpack('2I', packet[:8])
            packet = packet[8:]
            if tlvType == 1:
                results['detected_objects'] = self.parseDetectedObjects(packet, tlvLength)
            elif tlvType == 2:
                results['range_profile'] = self.parseRangeProfile(packet, tlvLength)
            elif tlvType == 6:
                results['stats'] = self.parseStats(packet, tlvLength)
            else:
                raise Exception("Unimplemented tlv type %d"%(tlvType))
        return results
    
    def run_once(self, is_file=False):
        if is_file:
            with open(self.data_port, 'rb') as f:
                data = f.read()
        else:
            data = self.data_port.read(self.data_port.in_waiting)
        packet_start_list = self.find_all_magic(data)
        if packet_start_list:
            packet_start_list.append(len(data))
            for st, end in zip(packet_start_list[:-1], packet_start_list[1:]):
                packet = data[st:end]
                results = self.parse_packet(packet)
                if results is not None:
                    self.data_queue.put(results)

    def run(self):
        while True:
            data = self.data_port.read(self.data_port.in_waiting)
            packet_start_list = self.find_all_magic(data)
            if not packet_start_list:
                continue
            else:
                packet_start_list.append(len(data))
                for st, end in zip(packet_start_list[:-1], packet_start_list[1:]):
                    packet = data[st:end]
                    results = self.parse_packet(packet)
                    if results is not None:
                        self.data_queue.put(results)
            time.sleep(0.1)
