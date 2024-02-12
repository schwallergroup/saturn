class RewardTracker:
    def __init__(self,
                 patience: int):
        self.patience = patience
        self.best_reward = None
        self.consecutive_improvements = 0
        self.beam_executions = 0

    def is_beam_epoch(self, reward: float):
        """
        this method returns a boolean whether to execute Beam Enumeration or not.
        The condition is if the reward increases for patience number of successive epochs to mitigate stochastic improvement.
        """
        # initialization
        if self.best_reward is None:
            self.best_reward = reward
            return False
        # reward improved
        elif reward > self.best_reward:
            self.consecutive_improvements += 1
            if self.consecutive_improvements == self.patience:
                self.best_reward = reward
                self.consecutive_improvements = 0
                self.beam_executions += 1
                return True
            return False
        # reward did not improve
        else:
            self.consecutive_improvements = 0
            return False
