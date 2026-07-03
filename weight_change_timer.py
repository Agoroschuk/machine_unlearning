import os
import time
import torch
from transformers import TrainerCallback

class WeightChangeTimerCallback(TrainerCallback):
    """
    Time measuring during forward + loss + backward + optimizer update + scheduler update inside training step

    forget.py: weight_change_timing_file
        ↓
    CustomFamilyTrainerForgetting: self.weight_change_timing_file
            ↓
    WeightChangeTimerCallback: self.output_file
            ↓
    creation of weight_change_seconds.tmp on google drive
    """
    def __init__(self, output_file, sync_cuda=True):
        self.output_file = output_file
        self.sync_cuda = sync_cuda
        self.weight_change_seconds = 0.0
        self._step_start = None

    def _sync(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_step_begin(self, args, state, control, **kwargs):
        self._sync()
        self._step_start = time.perf_counter() # approx. as time.time()
    
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
