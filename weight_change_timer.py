import os
import time
import torch
from transformers import TrainerCallback

class WeightChangeTimerCallback(TrainerCallback):
    """
    Измерение времени forward + loss + backward + optimizer update + scheduler update внутри training step

    forget.py: weight_change_timing_file
        ↓
    CustomFamilyTrainerForgetting: self.weight_change_timing_file
            ↓
    WeightChangeTimerCallback: self.output_file
            ↓
    weight_change_seconds.tmp на диске
    """
    def __init__(self, output_file, sync_cuda=True):
        self.output_file = output_file
        self.sync_cuda = sync_cuda
        self.weight_change_seconds = 0.0
        self._step_start = None

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            # нужно, чтобы не занизить время измерения этапов из-за асинхронности CUDA-операций, 
            # т.е. чтобы дождаться завершения стадии, измерить реальное время вычислений на gpu, а не время постановки в очередь
            torch.cuda.synchronize()

    def on_step_begin(self, args, state, control, **kwargs):
        self._sync()
        self._step_start = time.perf_counter() # аналог time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        self._sync()
        if self._step_start is not None:
            self.weight_change_seconds += time.perf_counter() - self._step_start
            self._step_start = None

    def on_train_end(self, args, state, control, **kwargs):
        if not args.should_save:
            return
    
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(self.output_file, "w") as f:
            f.write(f"{self.weight_change_seconds:.6f}\n")
