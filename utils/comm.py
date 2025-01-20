# from ..utils import logger

PAD_DICT = {}
import torch.distributed as dist
import torch

from torch.distributed import ProcessGroup

class ParallelManager():
    # 类变量，用于存储 DeepSpeed 引擎
    engine = None

    def __init__(self):
        pass

    @staticmethod
    def get_engine():
        """
        获取 DeepSpeed 引擎，如果未初始化则抛出异常
        """
        if ParallelManager.engine is None:
            raise RuntimeError("DeepSpeed engine has not been initialized.")
        return ParallelManager.engine    

    @staticmethod
    def set_engine(deepspeed_engine = None):
        ParallelManager.engine = deepspeed_engine

    @staticmethod
    def get_data_parallel_group():
        if ParallelManager.engine is None:
            raise RuntimeError("DeepSpeed engine has not been initialized.")
        return ParallelManager.engine.data_parallel_group

    @staticmethod
    def get_seq_parallel_group():
        if ParallelManager.engine is None:
            raise RuntimeError("DeepSpeed engine has not been initialized.")
        return ParallelManager.engine.seq_parallel_group        
    @staticmethod
    def is_enable_seq_parallel():
        if hasattr(ParallelManager.engine,"seq_parallel_group") and dist.get_world_size(ParallelManager.engine.seq_parallel_group) > 1:
            return True
        return False
    
    @staticmethod
    def get_seq_parallel_size():
        if ParallelManager.is_enable_seq_parallel():
            return dist.get_world_size(ParallelManager.engine.seq_parallel_group)
        return 0



def set_pad(name: str, dim_size: int, parallel_group: dist.ProcessGroup):
    sp_size = dist.get_world_size(parallel_group)
    # sp_size是一个组中的GPU数量，比如说=4，就是说明这个序列要被分到4个GPU上
    # 可能序列长度不能整除， % 得到的就是剩下的部分，需要pad补齐
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global PAD_DICT
    PAD_DICT[name] = pad


def get_pad(name) -> int:
    return PAD_DICT[name]



def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_pad: int = 0,
    gather_pad: int = 0,
):
    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    assert (
        input_.shape[scatter_dim] % dist.get_world_size(process_group) == 0
    ), f"Dimension to scatter ({input_.shape[scatter_dim]}) is not divisible by world size ({dist.get_world_size(process_group)})"
    input_ = _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)

    return input_







# ======================================================
# Sequence Gather & Split
# ======================================================

# 这个方法就完成了在一个序列并行组里，将长序列按照序列长度维度进行切分，并对不够的进行pad补齐

def _split_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    if pad > 0:
        pad_size = list(input_.shape)
        pad_size[dim] = pad
        input_ = torch.cat([input_, torch.zeros(pad_size, dtype=input_.dtype, device=input_.device)], dim=dim)

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, f"dim_size ({dim_size}) is not divisible by world_size ({world_size})"

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    # logger.info(f'[Global rank {dist.get_rank()}],[sequence_group rank]: {rank}')
    output = tensor_list[rank].contiguous()
    return output


def _gather_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type == "cuda"
    # 这是一个同步原语，每个GPU上的进程同时调用，把input填充到对应的tensor_list中
    # 最后每个GPU上都拿到完整的序列
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim)
    # logger.info(f'[Rank {dist.get_rank()}][before remove pad] output shape is {output.shape}, pad = {pad}')
    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)

    # logger.info(f'[Rank {dist.get_rank()}][after remove pad] output shape is {output.shape}, pad = {pad}')
    return output



class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Gather the input sequence.

    Args:
        input_: input matrix.
        process_group: process group.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad

        return _gather_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None




class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split sequence.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _split_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)
        print(f'[Rank {dist.get_rank()}]: ctx.pad = {ctx.pad}')    
        return _gather_sequence_func(grad_output, ctx.process_group, ctx.dim ,ctx.pad), None, None, None, None

def split_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, pad)



def gather_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, pad)





# ======================================================
# AlltoAll
# ======================================================


def _all_to_all_func(input_, world_size, group, scatter_dim, gather_dim):
    # 沿着scatter_dim维度进行切分，这里就是dimension维度
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)

        return _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)



def remove_padding(input_, process_group=None, padding_key='',dim=2):
    pad = get_pad(padding_key)
    if pad > 0:
        input_ = input_.narrow(dim, 0 , input_.size(dim) - pad) 

    return input_

def add_padding(input_, process_group = None, padding_key='',dim=1):
    pad = get_pad(padding_key)
    if pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[dim] = pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_,pad_tensor], dim = dim)
    return input_     



# remove extra_encoder and video padding
def remove_extra_encoder(input_, text_seq_length,  process_group = None,padding_key = '',dim = 2):
    sp_size = dist.get_world_size(process_group)
    split_seq = input_.split(int(input_.size(2) // sp_size), dim = dim)
    encoder_hidden_states = split_seq[0][:, :, -text_seq_length: ]
    new_seq = []

    for i in range(sp_size):
        new_seq.append(split_seq[i][:,:,:-text_seq_length])
    hidden_states = torch.cat(new_seq, dim = dim)

    pad = get_pad(padding_key)
    if pad > 0:
        hidden_states = hidden_states.narrow(dim,0, hidden_states.size(dim)-pad)
        # logger.info(f'[Rank {dist.get_rank()}][after remove pad] output shape is {input_.shape}, pad = {pad}')
        
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=dim)
    return hidden_states    

def add_extra_encoder(input_, text_seq_length,  process_group = None,padding_key = '',dim = 1):
    hidden_states = input_[:, :-text_seq_length, :]
    encoder = input_[:, -text_seq_length:, :]
    pad = get_pad(padding_key)
    if pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[1] = pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        hidden_states = torch.cat([hidden_states, pad_tensor], dim=1)
        input_ = torch.cat([hidden_states,encoder], dim=dim)
    sp_size = dist.get_world_size(process_group)
    seq = hidden_states.split( int(hidden_states.size(dim) // sp_size), dim=dim)
    new_seq = []
    
    for i in range(sp_size):
        new_seq.append(seq[i])
        new_seq.append(encoder)

    hidden_states = torch.cat(new_seq,dim = dim)
    return hidden_states    




