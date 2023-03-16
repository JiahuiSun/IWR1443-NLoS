"""
1. 解析的对不对？没问题
2. 
"""
import queue
from read_IWR1443 import read_IWR1443


data_port = "data/xwr14xx_processed_stream_2023_03_12T15_40_56_912.dat"
data_queue = queue.Queue()
read_thread = read_IWR1443(data_queue, data_port)
read_thread.run_once(is_file=True)
print(data_queue.qsize())
