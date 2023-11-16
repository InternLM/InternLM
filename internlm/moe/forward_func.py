import torch

from .communication import moe_all_to_all, moe_stream_acquire, moe_stream_release


def no_overlap_moe_forward(inputs, expert_fn, ep_group, ep_size, num_local_experts, d_model):
    """
    Preform moe forward computation sequentially.
    For example:
        alltoall(d)---->expert_fn(d)--->alltoall(d)
    """

    inputs = moe_all_to_all.apply(ep_group, inputs)

    # Re-shape after all-to-all: ecm -> gecm
    inputs = inputs.reshape(ep_size, num_local_experts, -1, d_model)
    expert_output = expert_fn(inputs)

    expert_output = moe_all_to_all.apply(ep_group, expert_output)

    return expert_output


def overlap_moe_forward(inputs, expert_fn, a2a_ffn_overlap_degree, ep_group, ep_size, num_local_experts, d_model):
    """
    Split the input based on a2a_ffn_overlap_degree and then execute the alltoall and experts function
    on different stream to overlap the communication and computation cost.
    For example:
        communication stream:  alltoall(d[0])---->alltoall(d[1])---->alltoall(d[0])---->alltoall(d[1])
        computation stream:               expert_fn(d[0])  ---->  expert_fn(d[1])

    """

    # inputs shape: (e,c,m). split the inputs on 'c' dimension
    input_chunks = inputs.chunk(a2a_ffn_overlap_degree, dim=1)

    expert_inputs = [None for _ in range(a2a_ffn_overlap_degree)]
    expert_outputs = [None for _ in range(a2a_ffn_overlap_degree)]

    ready_events = [torch.cuda.Event() for _ in range(a2a_ffn_overlap_degree)]
    alltoall_stream = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(a2a_ffn_overlap_degree)]
    experts_stream = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(a2a_ffn_overlap_degree)]

    # NOTE: async alltoall seems unable to improve the performance
    # first all2all, execute on alltoall streams
    for i, input_split in enumerate(input_chunks):
        moe_stream_release.apply(torch.cuda.default_stream(), ready_events[i])

        moe_stream_acquire.apply(alltoall_stream[i], ready_events[i])
        expert_inputs[i] = moe_all_to_all.apply(ep_group, input_split)
        moe_stream_release.apply(alltoall_stream[i], ready_events[i])

    # expert function, execute on experts stream
    for i in range(a2a_ffn_overlap_degree):
        moe_stream_acquire.apply(experts_stream[i], ready_events[i])
        # Re-shape after all-to-all: ecm -> gecm
        expert_inputs[i] = expert_inputs[i].reshape(ep_size, num_local_experts, -1, d_model)
        expert_outputs[i] = expert_fn(expert_inputs[i])
        moe_stream_release.apply(experts_stream[i], ready_events[i])

    # second all2all, execute on alltoall streams
    for i in range(a2a_ffn_overlap_degree):
        moe_stream_acquire.apply(alltoall_stream[i], ready_events[i])
        expert_outputs[i] = moe_all_to_all.apply(ep_group, expert_outputs[i])
        moe_stream_release.apply(alltoall_stream[i], ready_events[i])

        moe_stream_acquire.apply(torch.cuda.default_stream(), ready_events[i])

    # expert_outputs shape: (g, e,c,m). cat the outputs on 'c' dimension
    expert_output_gathered = torch.cat(expert_outputs, dim=2)

    return expert_output_gathered
