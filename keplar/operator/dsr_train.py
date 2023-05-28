
from dsr.dso.train import Trainer

class dsr_Train(Trainer):
    def __init__(self, sess, policy, policy_optimizer, gp_controller, logger, pool, **kwargs):
        Trainer.__init__(self, sess, policy, policy_optimizer, gp_controller, logger, pool, **kwargs)

    def run_one_step(self,override=None):
        Trainer.run_one_step(self,override=None)
        # pass

        

    

    