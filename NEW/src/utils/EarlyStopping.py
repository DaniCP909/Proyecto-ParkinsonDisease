class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.stop = False
        self.best_state = None

    def step(self, val_loss, model):
        print(f"[EarlyStopping] val_loss={val_loss:.6f}, best_loss={self.best_loss:.6f}, counter={self.counter}")
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            print(f"[EarlyStopping] No mejora. Counter={self.counter}")
            if self.counter >= self.patience:
                self.stop = True
                print(f"[EarlyStopping] Stop triggered!")
