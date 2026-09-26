"""Single-process specialist model parallelism, with device-local optimizer state."""
import torch


def layout(cfg, devices):
    devices = tuple(devices)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError('provide unique devices')
    if len(devices) > 1 and cfg.stacks.uses_io_projection:
        raise ValueError('multi-GPU placement currently requires equal core and stack widths')
    sizes = [cfg.stacks.params_for_stack(i) for i in range(cfg.stacks.n_stacks)]
    loads = [cfg.n_params - sum(sizes)] + [0] * (len(devices)-1)
    assignments = [0] * len(sizes)
    for index in sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True):
        device = min(range(len(devices)), key=lambda i: loads[i])
        assignments[index] = device
        loads[device] += sizes[index]
    return {'devices': devices, 'stack_devices': [devices[i] for i in assignments],
            'parameters_per_device': dict(zip(devices, loads))}


def place_model(model, devices):
    """Place modules directly, never first load the entire model onto GPU 0.

    Configure before constructing an optimizer. All shared components and the
    optional bank IO projections stay on the primary device.
    """
    plan = layout(model.cfg, devices)
    primary = devices[0]
    if len(devices) > 1 and any(not str(d).startswith('cuda:') for d in devices):
        raise ValueError('multi-device placement requires explicit CUDA indices')
    for name in ('codecs', 'core', 'router', 'context_memory', 'rope'):
        module = getattr(model, name, None)
        if module is not None:
            module.to(primary)
    for name in ('entry', 'exit'):
        module = getattr(model.bank, name)
        if module is not None:
            module.to(primary)
    if model.bank_gate is not None:
        model.bank_gate.data = model.bank_gate.data.to(primary)
    for stack, device in zip(model.bank.stacks, plan['stack_devices']):
        stack.to(device)
    model.device_plan = plan
    return plan


def native_bf16(devices):
    """Native bf16 on every device, decided per vendor rather than per number.

    The obvious version -- ``get_device_capability(d)[0] >= 8`` -- is an NVIDIA
    SM major-version test, and it is the *right* test on NVIDIA: bf16 arrives
    with Ampere (sm_80), and Turing and Pascal only emulate it, so training in
    bf16 there is quietly slower than fp32 rather than faster.

    On a ROCm build it is not a test at all. ``get_device_capability`` still
    returns a two-tuple, but it is derived from the HIP arch, not from an
    NVIDIA compute capability, and comparing its first element against 8 asks a
    question about AMD hardware that has no meaning. It can fail in both
    directions -- refusing bf16 on an MI300 that has it in silicon, or granting
    it on an older RDNA part that does not -- and both failures are silent: the
    first is merely slow, the second produces a run whose numerics are emulated
    and whose loss curve looks unremarkable.

    ``runtime.device.detect`` already decides this correctly by matching the
    gfx arch string, so the fix is to ask it rather than to re-derive it here.
    Two implementations of one predicate is how they came apart in the first
    place.
    """
    if not devices:
        return False
    if not all(str(d).startswith('cuda') for d in devices):
        return False
    from .device import detect
    # Query every selected device. detect() without an index always inspects
    # GPU 0, which is wrong when a single training worker runs on cuda:1.
    return all(detect(str(device)).bf16 for device in devices)


def memory_snapshot(devices):
    return {str(d): {'allocated_gb': torch.cuda.memory_allocated(d)/1e9,
                     'reserved_gb': torch.cuda.memory_reserved(d)/1e9,
                     'peak_gb': torch.cuda.max_memory_allocated(d)/1e9}
            for d in devices if str(d).startswith('cuda')}
