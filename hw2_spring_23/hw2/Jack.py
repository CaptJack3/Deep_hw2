def fit(
        self,
        dl_train: DataLoader,
        dl_test: DataLoader,
        num_epochs: int,
        checkpoints: str = None,
        early_stopping: int = None,
        print_every: int = 1,
        **kw,
) -> FitResult:
    """
    Trains the model for multiple epochs with a given training set,
    and calculates validation loss over a given validation set.
    :param dl_train: Dataloader for the training set.
    :param dl_test: Dataloader for the test set.
    :param num_epochs: Number of epochs to train for.
    :param checkpoints: Whether to save model to file every time the
        test set accuracy improves. Should be a string containing a
        filename without extension.
    :param early_stopping: Whether to stop training early if there is no
        test loss improvement for this number of epochs.
    :param print_every: Print progress every this number of epochs.
    :return: A FitResult object containing train and test losses per epoch.
    """

    actual_num_epochs = 0
    epochs_without_improvement = 0

    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    best_acc = None

    for epoch in range(num_epochs):
        verbose = False  # pass this to train/test_epoch.
        if print_every > 0 and (
                epoch % print_every == 0 or epoch == num_epochs - 1
        ):
            verbose = True
        self._print(f"--- EPOCH {epoch + 1}/{num_epochs} ---", verbose)

        # TODO: Train & evaluate for one epoch
        #  - Use the train/test_epoch methods.
        #  - Save losses and accuracies in the lists above.
        # ====== YOUR CODE: ======
        e_train_losses, e_train_acc = self.train_epoch(dl_train, verbose=verbose, **kw)
        e_train_avg_loss = sum(e_train_losses) / len(e_train_losses)
        train_loss.append(e_train_avg_loss)
        train_acc.append(e_train_acc)

        e_test_losses, e_test_acc = self.test_epoch(dl_test, verbose=verbose, **kw)
        e_test_avg_loss = sum(e_test_losses) / len(e_test_losses)
        test_loss.append(e_test_avg_loss)
        test_acc.append(e_test_acc)
        # ========================

        # TODO:
        #  - Optional: Implement early stopping. This is a very useful and
        #    simple regularization technique that is highly recommended.
        #  - Optional: Implement checkpoints. You can use the save_checkpoint
        #    method on this class to save the model to the file specified by
        #    the checkpoints argument.
        if best_acc is None or e_test_acc > best_acc:
            # ===== = YOUR CODE: ======
            if checkpoints is not None:
                self.save_checkpoint(checkpoints)
            epochs_without_improvement = 0
            # ========================
        else:
            # ====== YOUR CODE: ======
            epochs_without_improvement += 1
            if not (early_stopping is None) and epochs_without_improvement >= early_stopping:
                break
            # ========================

    return FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)