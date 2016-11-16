from tools import pp, array_pad
from GPU_availability import GPU_availability as GPU
from data_utils import *
import os

def define_gpu(num):
    gpu_list = GPU()[:num]
    gpu_str  = ','.join( map(str, gpu_list) )
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_str 
    return gpu_str
