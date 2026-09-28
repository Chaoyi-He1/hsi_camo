import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TrainLogger(object):
    '''
    Fan every scalar/histogram/image out to TensorBoard (runs/<name>) and Weights & Biases from one place,
    main process only. W&B: entity/project from args; any wandb error (no entity, no network) falls back to
    offline mode so training never blocks; `wandb sync wandb/offline-run-*` pushes those runs later.
    '''

    def __init__(self, args, cfg):
        self.enabled = int(getattr(args, 'rank', 0)) == 0
        self.tb = None
        self.wandb_run = None
        self.mode = 'disabled'
        self._defined = set()            # W&B metric groups whose x-axis has been declared (see _wandb_log)
        if not self.enabled:
            return
        self.tb = SummaryWriter(log_dir=os.path.join(getattr(args, 'runs_dir', 'runs'), args.name))
        if getattr(args, 'wandb', False):
            import wandb
            config = {**{k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in vars(args).items()}, **cfg}
            common = dict(entity=args.wandb_entity, project=args.wandb_project, name=args.name, config=config,
                          dir=getattr(args, 'wandb_dir', 'wandb'))
            os.makedirs(common['dir'], exist_ok=True)
            for mode in (None, 'offline'):
                try:
                    self.wandb_run = wandb.init(mode=mode, **common) if mode else wandb.init(**common)
                    self.mode = mode or 'online'
                    break
                except Exception as e:                                    # CommError, UsageError, network errors
                    print(f"wandb.init({mode or 'online'}) failed: {type(e).__name__}: {str(e)[:120]}")
            if self.mode == 'offline':
                print(f"W&B running offline; sync later with: wandb sync {common['dir']}/wandb/offline-run-*")

    def _wandb_log(self, tag, value, step):
        '''
        W&B's built-in step must increase monotonically, so per-step 'train/*' logging followed by per-epoch
        'val/*' logging under step= silently drops the epoch metrics. Instead each metric group gets its own
        x-axis, declared once via define_metric: 'train/*' is plotted against global_step, everything else
        against epoch, and the axis value travels inside the logged dict.
        '''
        group = tag.split('/', 1)[0]
        axis = 'global_step' if group == 'train' else 'epoch'
        if group not in self._defined:
            self.wandb_run.define_metric(f'{group}/*' if '/' in tag else tag, step_metric=axis)
            self._defined.add(group)
        self.wandb_run.log({tag: value, axis: int(step)})

    def scalar(self, tag, value, step):
        if not self.enabled:
            return
        value = float(value)
        self.tb.add_scalar(tag, value, step)
        if self.wandb_run is not None:
            self._wandb_log(tag, value, step)

    def scalars(self, values, step, prefix=''):
        for k, v in values.items():
            if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v)):   # np.float32 too
                self.scalar(prefix + k, v, step)

    def histogram(self, tag, values, step):
        if not self.enabled:
            return
        v = np.asarray(values, dtype=np.float32).ravel()
        self.tb.add_histogram(tag, v, step)
        if self.wandb_run is not None:
            import wandb
            self._wandb_log(tag, wandb.Histogram(v), step)

    def image(self, tag, hwc_uint8, step):
        if not self.enabled:
            return
        img = np.asarray(hwc_uint8)
        self.tb.add_image(tag, img, step, dataformats='HWC')
        if self.wandb_run is not None:
            import wandb
            self._wandb_log(tag, wandb.Image(img), step)

    def table(self, tag, columns, rows, step):
        if self.enabled and self.wandb_run is not None:
            import wandb
            self._wandb_log(tag, wandb.Table(columns=list(columns), data=[list(r) for r in rows]), step)

    def finish(self):
        if self.tb is not None:
            self.tb.flush(); self.tb.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
