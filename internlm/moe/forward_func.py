import torch

from .communication import moe_all_to_all, moe_stream_acquire, moe_stream_release

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == "s,se->se":
        # [1, s] * [s, e]
        return a.reshape(a.shape[0], -1) * b
    elif rule == "se,sc->sec":
        # [s,e,1] * [s,1,c]
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == "se,se->s":
        # [s,1,e] * [s,e,1]
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == "sec,sm->ecm":
        # [e*c, s] * [s, m]
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == "sec,ecm->sm":
        # [s, e*c] * [e*c, m]
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == "ks,ksm->sm":
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)


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


def overlap_moe_forward(
    reshaped_inputs, gata_fn, expert_fn, a2a_ffn_overlap_degree, ep_group, ep_size, num_local_experts, d_model
):
    """
    Split the input based on a2a_ffn_overlap_degree and then execute the alltoall and experts function
    on different stream to overlap the communication and computation cost.
    For example:
        communication stream:  alltoall(d[0])---->alltoall(d[1])---->alltoall(d[0])---->alltoall(d[1])
        computation stream:               expert_fn(d[0])  ---->  expert_fn(d[1])

    """

    # variables for stream control
    ready_events = [torch.cuda.Event() for _ in range(a2a_ffn_overlap_degree)]
    alltoall_stream = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(a2a_ffn_overlap_degree)]
    experts_stream = [torch.cuda.Stream(torch.cuda.current_device()) for _ in range(a2a_ffn_overlap_degree)]

    # local variables for gating and expert computing
    l_aux = torch.tensor(0.0, dtype=reshaped_inputs.dtype, device=reshaped_inputs.device)
    dispatched_inputs = [None for _ in range(a2a_ffn_overlap_degree)]
    expert_inputs = [None for _ in range(a2a_ffn_overlap_degree)]
    expert_outputs = [None for _ in range(a2a_ffn_overlap_degree)]
    combine_weights = [None for _ in range(a2a_ffn_overlap_degree)]
    combined_output = [None for _ in range(a2a_ffn_overlap_degree)]

    # (s,d), split by "s" dimension
    input_chunks = reshaped_inputs.chunk(a2a_ffn_overlap_degree, dim=0)

    # gating computing
    for i, input_split in enumerate(input_chunks):
        moe_stream_release.apply(torch.cuda.default_stream(), ready_events[i])

        moe_stream_acquire.apply(experts_stream[i], ready_events[i])
        cur_l_aux, combine_weights[i], dispatch_mask, exp_counts = gata_fn(input_split)
        dispatched_inputs[i] = einsum(
            "sec,sm->ecm", dispatch_mask.type_as(input_split), input_split
        )  # TODO: heavy memory usage due to long sequence length
        l_aux += cur_l_aux
        moe_stream_release.apply(experts_stream[i], ready_events[i])

        # NOTE: async alltoall seems unable to improve the performance
        # first all2all, execute on alltoall streams
        moe_stream_acquire.apply(alltoall_stream[i], ready_events[i])
        expert_inputs[i] = moe_all_to_all.apply(ep_group, dispatched_inputs[i])
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

    for i in range(a2a_ffn_overlap_degree):
        moe_stream_acquire.apply(experts_stream[i], ready_events[i])
        # Re-shape back: gecm -> ecm
        expert_outputs[i] = expert_outputs[i].reshape(ep_size * num_local_experts, -1, d_model)
        combined_output[i] = einsum(
            "sec,ecm->sm", combine_weights[i].type_as(input_chunks[0]), expert_outputs[i].type_as(input_chunks[0])
        )
        moe_stream_release.apply(experts_stream[i], ready_events[i])

        moe_stream_acquire.apply(torch.cuda.default_stream(), ready_events[i])

    combined_output = torch.cat(combined_output)
    return combined_output, l_aux / a2a_ffn_overlap_degree, exp_counts
