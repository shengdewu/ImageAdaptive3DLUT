from engine.schedule.scheduler import BaseScheduler
from trainer.AdaptiveTrainer import AdaptiveTrainer


class AdaptiveScheduler(BaseScheduler):
    def __init__(self):
        super(AdaptiveScheduler, self).__init__()
        return

    def lunch_func(self, cfg, args):
        trainer = AdaptiveTrainer(cfg)
        trainer.resume_or_load(args.resume)
        trainer.loop()
        return
