#hyperparameter-Tuning here
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def objective(trial: optuna.trial.Trial) -> float:
    
    # We optimize the number of layers, hidden units in each layer and dropouts.
    data_augmentation = trial.suggest_categorical("data_augmentation", [True, False])
    n_layers = trial.suggest_int("n_layers", 4, 6)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    attn_dropout = trial.suggest_float("attn_dropout", 0.0, 0.5)
    encoder_channels = trial.suggest_categorical("encoder_channels", [128,256,512])
    k= trial.suggest_categorical("k", [16,32,64,128])
   
    
    train_dataloader = DataLoader(ModelNet40(1024,partition='train', data_augmentation=True), num_workers=8, batch_size=loader_batch, shuffle=True, drop_last=True)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=15)
    model = PCT_Classifier_4(40, learning_rate=0.0001, attention_layers=n_layers, dropout= dropout, attention_dropout=attn_dropout,
                             encoder_channels= encoder_channels, k=k)

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=150,
        gpus=-1,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_accuracy"), early_stopping_callback],
    )
    hyperparameters = dict( data_augmentation=data_augmentation, n_layers=n_layers, dropout=dropout, attn_dropout=attn_dropout, encoder_channels=encoder_channels, k=k)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model, train_dataloader, val_dataloader)
    del model
    torch.cuda.empty_cache()
    return trainer.callback_metrics["val_accuracy"].item()
