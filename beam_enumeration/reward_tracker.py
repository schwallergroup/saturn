class RewardTracker:
    """
    This class is used to track the mean reward to decide whether to execute Beam Enumeration or not, at a given generation epoch.
    """
    def __init__(self,
                 patience: int):
        self.patience = patience
        self.best_reward = None
        self.consecutive_improvements = 0
        self.beam_executions = 0

    def is_beam_epoch(self, mean_reward: float) -> bool:
        """
        This method returns a boolean whether to execute Beam Enumeration or not.
        The condition is if the mean reward increases for patience number of successive epochs (to mitigate stochastic improvement).
        """
        # Initialization
        if self.best_reward is None:
            self.best_reward = mean_reward
            return False
        # Reward improved
        elif mean_reward > self.best_reward:
            self.consecutive_improvements += 1
            if self.consecutive_improvements == self.patience:
                self.best_reward = mean_reward
                self.consecutive_improvements = 0
                self.beam_executions += 1
                return True
            return False
        # Reward did not improve
        else:
            self.consecutive_improvements = 0
            return False
