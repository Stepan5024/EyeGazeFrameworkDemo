from torch.optim import lr_scheduler

class LRScheduler:
    """
    Check if the validation loss does not decrease for a given number of epochs 
    (patience), then decrease the learning rate by a given 'factor'.
    """

    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the learning rate
        :param min_lr: least learning rate value to reduce to while updating
        :param factor: factor by which the learning rate should be updated
        :returns: new learning rate = old learning rate * factor
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=self.patience,
            factor=self.factor, min_lr=self.min_lr, verbose=True
        )
    def __call__(self, validation_loss):
        self.lr_scheduler.step(validation_loss)

class EarlyStopping:
    """
    Early stopping breaks the training procedure when the loss does not improve 
    over a certain number of iterations.
    """

    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: number of epochs to wait before stopping the training procedure
        :param min_delta: the minimum difference between previous and the new loss 
                          required to consider the network is improving.
        """
        self.early_stop_enabled = False
        self.min_delta = min_delta
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def __call__(self, validation_loss):
        # Update the validation loss if the condition doesn't hold
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif (self.best_loss - validation_loss) < self.min_delta:
            self.counter += 1
            print(f"[INFO] Early stopping: {self.counter}/{self.patience}...\n")
            if self.counter >= self.patience:
                self.early_stop_enabled = True
                print("[INFO] Early stopping enabled")
        elif (self.best_loss - validation_loss) > self.min_delta:
            # Reset the early stopping counter
            self.counter = 0
            self.best_loss = validation_loss

