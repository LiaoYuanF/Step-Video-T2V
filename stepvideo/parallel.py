import torch.distributed as dist
import xfuser
import torch
import os


def initialize_parall_group(ring_degree, ulysses_degree, tensor_parallel_degree):
    # 强制设置当前设备（Ray环境下永远用逻辑ID 0）
    torch.cuda.set_device(0)
    
    # 显式从环境变量获取参数
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # 初始化进程组（必须显式指定参数）
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    
    # 后续初始化必须放在进程组初始化之后
    xfuser.core.distributed.init_distributed_environment(
        rank=rank, 
        world_size=world_size
    )
    
    xfuser.core.distributed.initialize_model_parallel(
        sequence_parallel_degree=ulysses_degree,
        ring_degree=ring_degree,
        ulysses_degree=ulysses_degree,
        tensor_parallel_degree=tensor_parallel_degree,
    )

    # 二次确认设备设置
    torch.cuda.set_device(0)  # 关键！不要使用dist.get_rank()


def get_parallel_group():
    return xfuser.core.distributed.get_world_group()

def get_sequence_parallel_world_size():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()

def get_sequence_parallel_rank():
    return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()

def get_sp_group():
    return xfuser.core.distributed.parallel_state.get_sp_group()



def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs['parallel']:            
            hidden_states = torch.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
            kwargs['attn_mask'] = torch.chunk(kwargs['attn_mask'], get_sequence_parallel_world_size(), dim=-2)[get_sequence_parallel_rank()]
        output = fn_(_, hidden_states, *args, **kwargs)
        
        if kwargs['parallel']:
            output = get_sp_group().all_gather(output.contiguous(), dim=-2)
        
        return output
     
    return wrapTheFunction