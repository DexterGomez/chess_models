import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import time
import mlflow
import os

from functools import partial

from torch.utils.data import DataLoader

from src.data_loader import ChessDataset, AsyncDataPrefetcher
from src.mlx_model import ChessNet


def main():
    params = {
        "BATCH_SIZE": 2048,
        "LEARNING_RATE": 1e-3,
        "EPOCHS": 10,
        "H5_FILE": "data_silver/dataset_ajedrez_final.h5",
        "FRAMEWORK": "MLX",
        "NUM_WORKERS": 4,
        "PREFETCH_FACTOR": 4,
        "PREFETCH_QUEUE_SIZE": 4,
    }

    print(f"MLX Device: {mx.default_device()}")

    # initialize data loader
    dataset = ChessDataset(params["H5_FILE"])
    dataloader = DataLoader(
        dataset,
        batch_size=params["BATCH_SIZE"],
        shuffle=True,
        num_workers=params["NUM_WORKERS"],
        prefetch_factor=params["PREFETCH_FACTOR"],
        drop_last=True,
        persistent_workers=True,
    )

    # model and optimizer
    model = ChessNet()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=params["LEARNING_RATE"])
    model.train(True)

    # loss function
    # since we have both policy and value heads we use this 
    # custom loss to minize the sum of both head's errors
    def loss_fn(model, x, y_policy, y_value):
        p_logits, v_pred = model(x)
        loss_p = nn.losses.cross_entropy(p_logits, y_policy, reduction='mean')
        loss_v = nn.losses.mse_loss(v_pred.flatten(), y_value.flatten(), reduction='mean')
        return loss_p + loss_v, (loss_p, loss_v)

    # compilation with state tracking for performance:
    # defined states to use, telling the compiler that the input and output
    # structure of the graph stays the same so it does not recompile it every time
    state = [model.state, optimizer.state]
    # mx.compile compiles the function into the computation graph
    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y_policy, y_value):
        # calculate the gradients
        (loss, (l_p, l_v)), grads = nn.value_and_grad(model, loss_fn)(model, x, y_policy, y_value)
        # update the weights
        optimizer.update(model, grads)
        return loss, l_p, l_v

    mlflow.set_experiment("Chess_Training_MLX")

    print("\nStarting training...")
    print(f"\tDataset size: {len(dataset):,} samples")
    print(f"\tTrainable parameters: {sum(v.size for _, v in tree_flatten(model.trainable_parameters()))}")
    print(f"\tBatches/epoch: {len(dataloader):,}")

    with mlflow.start_run():
        mlflow.log_params(params)
        steps_per_epoch = len(dataloader)

        for epoch in range(params["EPOCHS"]):
            epoch_start_time = time.time()
            total_loss_acc = 0.0
            total_p_loss_acc = 0.0
            total_v_loss_acc = 0.0
            num_logged = 0
            
            # this starts to read data from storage while GPU is bussy
            prefetcher = AsyncDataPrefetcher(
                dataloader, 
                queue_size=params["PREFETCH_QUEUE_SIZE"]
            )

            # iterates throught the batches
            for batch_idx, (x, y_policy, y_value) in enumerate(prefetcher):
                
                # executes the training logic for this batch
                # it returns the computation graphs nodes for the losses,
                # notthe values, just the 'instructions'
                loss, l_p, l_v = train_step(x, y_policy, y_value)
                
                # log and syncronize
                if batch_idx % 100 == 0:
                    
                    # until this point, MLX has only build the 'instructions'
                    # for calculations from the previus batches. Now here it
                    # calculates the loss with the GPU for the batches graphs
                    mx.eval(loss, l_p, l_v)

                    # this to get the actual values
                    loss_val = loss.item()
                    l_p_val = l_p.item()
                    l_v_val = l_v.item()
                    
                    total_loss_acc += loss_val
                    total_p_loss_acc += l_p_val
                    total_v_loss_acc += l_v_val
                    num_logged += 1
                    
                    # from here to the end of the loop is logging
                    elapsed = time.time() - epoch_start_time
                    samples_done = (batch_idx + 1) * params["BATCH_SIZE"]
                    speed = samples_done / elapsed if elapsed > 0 else 0
                    
                    print(
                        f"Ep {epoch+1} | {batch_idx}/{steps_per_epoch} | "
                        f"Loss: {loss_val:.4f} (P: {l_p_val:.4f}, V: {l_v_val:.4f}) | "
                        f"Speed: {speed:.0f} samp/s"
                    )
                    mlflow.log_metrics(
                        {
                            "batch_loss": loss_val,
                            "p_loss": l_p_val,
                            "v_loss": l_v_val,
                            "speed_samples_per_sec": speed,
                        },
                        step=epoch * steps_per_epoch + batch_idx
                    )

            # once all batches are processed, we use a last eval to ensure 
            # any pending updates are not left.
            mx.eval(model.parameters(), optimizer.state)
            
            epoch_duration = time.time() - epoch_start_time
            samples_per_sec = len(dataset) / epoch_duration
            avg_loss = total_loss_acc / max(num_logged, 1)
            avg_p_loss = total_p_loss_acc / max(num_logged, 1)
            avg_v_loss = total_v_loss_acc / max(num_logged, 1)

            print(f"\n{'='*60}")
            print(f"END EPOCH {epoch+1}/{params['EPOCHS']}")
            print(f"{'='*60}")
            print(f"    Avg Loss:   {avg_loss:.4f}")
            print(f"    Avg P Loss: {avg_p_loss:.4f}")
            print(f"    Avg V Loss: {avg_v_loss:.4f}")
            print(f"    Duration:   {epoch_duration:.1f}s")
            print(f"    Speed:      {samples_per_sec:.0f} samples/s")
            print(f"{'='*60}\n")

            mlflow.log_metrics(
                {
                    "epoch_avg_loss": avg_loss,
                    "epoch_avg_p_loss": avg_p_loss,
                    "epoch_avg_v_loss": avg_v_loss,
                    "epoch_duration_sec": epoch_duration,
                    "samples_per_sec": samples_per_sec
                },
                step=epoch + 1
            )

            # save checkpoints for the current epoch
            os.makedirs("checkpoints", exist_ok=True)
            checkpoint_path = f"checkpoints/chess_model_epoch_{epoch+1}.safetensors"
            model.save_weights(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # traininng complete and saving model weights
        final_model_path = "checkpoints/final_model.safetensors"
        model.save_weights(final_model_path)
        mlflow.log_artifact(final_model_path)
        print(f"\nTraining complete! Final model saved: {final_model_path}")


if __name__ == "__main__":
    main()