import os, platform, threading, time
import numpy as np
import torch


"""
Compute accounting: hardware info, wall-clock, peak GPU memory, GPU energy, FLOPs, parameters.

Energy uses NVML (pip install nvidia-ml-py). two estimates are recorded when available:
    - counter: nvmlDeviceGetTotalEnergyConsumption (Volta and newer, ex: T4, not P100)
    - sampled: board power polled every `interval_s` seconds, integrated with the trapezoid rule

    Nothing in this module is allowed to crash a training run; failures are recorded as null.
"""


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Stopwatch:
    """Wall-clock timer that synchronizes CUDA so GPU work is attributed correctly"""

    def __init__(self):
        self.t0 = None

    def start(self):
        _cuda_sync()
        self.t0 = time.perf_counter()
        return self

    def stop(self):
        _cuda_sync()
        return time.perf_counter() - self.t0


def _nvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def _nvml_handle(pynvml, device_index=0):
    """
    NVML ignores CUDA_VISIBLE_DEVICES, so match the CUDA device by UUID first,
    then fall back to the physical index named in CUDA_VISIBLE_DEVICES.
    """

    try:
        uuid = getattr(torch.cuda.get_device_properties(device_index), "uuid", None)

        if uuid is not None:
            s = str(uuid)
            s = s if s.startswith("GPU-") else "GPU-" + s

            for arg in (s, s.encode()):
                try:
                    return pynvml.nvmlDeviceGetHandleByUUID(arg)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        physical = int(visible.split(",")[device_index]) if visible else device_index
        return pynvml.nvmlDeviceGetHandleByIndex(physical)
    except Exception:
        return None


class EnergyMeter:
    def __init__(self, device_index=0, interval_s=0.2):
        self.device_index = device_index
        self.interval_s = interval_s
        self.pynvml = None
        self.handle = None
        self.samples = []
        self.counter_start_mj = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if not torch.cuda.is_available():
            return self

        self.pynvml = _nvml()

        if self.pynvml is None:
            return self

        self.handle = _nvml_handle(self.pynvml, self.device_index)

        if self.handle is None:
            return self

        try:
            self.counter_start_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:
            self.counter_start_mj = None

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                watts = self.pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
                self.samples.append((time.perf_counter(), watts))
            except Exception:
                pass

            self._stop.wait(self.interval_s)

    def stop(self):
        out = {
            "energy_j": None,
            "energy_method": None,
            "energy_counter_j": None,
            "energy_sampled_j": None,
            "mean_power_w": None,
            "max_power_w": None,
            "n_power_samples": len(self.samples),
        }

        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)

        if self.handle is not None and self.counter_start_mj is not None:
            try:
                end_mj = self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                out["energy_counter_j"] = (end_mj - self.counter_start_mj) / 1000.0
            except Exception:
                pass

        if len(self.samples) >= 2:
            t = np.array([s[0] for s in self.samples])
            w = np.array([s[1] for s in self.samples])
            out["energy_sampled_j"] = float(np.trapezoid(w, t)) if hasattr(np, "trapezoid") else float(np.trapz(w, t))
            out["mean_power_w"] = float(w.mean())
            out["max_power_w"] = float(w.max())
            out["n_power_samples"] = int(len(w))

        if out["energy_counter_j"] is not None:
            out["energy_j"], out["energy_method"] = out["energy_counter_j"], "nvml_counter"
        elif out["energy_sampled_j"] is not None:
            out["energy_j"], out["energy_method"] = out["energy_sampled_j"], f"nvml_power_sampling_{self.interval_s}s"

        return out


class RunCompute:
    """
    one per training run:
        rc = RunCompute().start()
        ... rc.add("train_s", seconds) ...
        summary = rc.finish()
    """

    def __init__(self, device_index=0, energy_interval_s=0.2):
        self.meter = EnergyMeter(device_index=device_index, interval_s=energy_interval_s)
        self.totals = {}
        self.watch = Stopwatch()

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self.meter.start()
        self.watch.start()
        return self

    def add(self, key, seconds):
        self.totals[key] = self.totals.get(key, 0.0) + float(seconds)

    def finish(self):
        out = {"wall_clock_s": self.watch.stop()}
        out.update({k: v for k, v in self.totals.items()})

        if torch.cuda.is_available():
            out["peak_memory_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            out["peak_memory_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20

        out.update(self.meter.stop())
        return out


def hardware_info():
    info = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    try:
        with open("/proc/meminfo") as f:
            kb = int(f.readline().split()[1])
            info["ram_gb"] = round(kb / 2**20, 1)
    except Exception:
        info["ram_gb"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
        info["cuda_version"] = torch.version.cuda
        info["n_visible_gpus"] = torch.cuda.device_count()

        pynvml = _nvml()

        if pynvml is not None:
            try:
                drv = pynvml.nvmlSystemGetDriverVersion()
                info["driver"] = drv.decode() if isinstance(drv, bytes) else drv
            except Exception:
                pass

    return info


def count_params(module, trainable_only=False):
    return int(sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)))


@torch.no_grad()
def forward_flops(module, *example_inputs):
    """
    FLOPs (= 2 x multiply-accumulates) of one forward pass, counted by torch's FlopCounterMode.
    run in eval mode with a batch of 2 (BatchNorm needs > 1 sample) and divided by 2.
    returns None if counting fails.
    """

    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None

    was_training = module.training
    module.eval()

    try:
        with FlopCounterMode(display=False) as counter:
            module(*example_inputs)
        total = counter.get_total_flops()
        batch = example_inputs[0].shape[0]
        return float(total) / batch
    except Exception:
        return None
    finally:
        module.train(was_training)
