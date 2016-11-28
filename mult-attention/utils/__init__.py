from GPU_availability import GPU_availability as GPU
from data_utils import *
from mdu import *
import pprint
import os
pp = pprint.PrettyPrinter()

def define_gpu(num):
    gpu_list = GPU()[:num]
    gpu_str = ','.join(map(str, gpu_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    return gpu_list


