import torch
from torch.utils.data import DataLoader
import os
import sys
from src.tee import Tee
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from PIL import Image
from skimage import filters


def train(
    dataset,
    criterion,
    optimizer,
    model,
    model_name,
    train_flag,
    test_flag,
    n_th_frame,
    future_f,
    num_epochs=1000,
    batch_size=25,
):
    checkpoint_file = "model/" + model_name + "/model_checkpoint.pth"
    if not os.path.exists("model/" + model_name):
        os.makedirs("model/" + model_name)
    f = open("model/" + model_name + "/log.txt", "w")
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)

    train_loader = DataLoader(
        dataset=dataset.train_dataset, batch_size=batch_size, shuffle=False
    )
    validation_loader = DataLoader(
        dataset=dataset.validation_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        dataset=dataset.test_dataset, batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_epoch = 0
    total_steps = 0

    if os.path.isfile(checkpoint_file):
        print("Loading saved model...")
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]

    # print(f"Model summary : {summary(model, (in_channels, in_seq_len))}")
    # torchinfo.summary(model, (in_channels, 10, 100), device="cpu")
    print(model)

    if train_flag:
        # Define early stopping parameters
        print("Starting training...")
        patience = 40  # Number of consecutive epochs without improvement
        best_val_loss = float("inf")
        consecutive_no_improvement = 0
        for epoch in range(current_epoch, num_epochs):
            train_loss = 0.0
            train_miou = 0.0

            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                y_hat = model(images)

                if n_th_frame is False:
                    label_arrays = np.split(labels[1].detach().numpy(), future_f)
                    y_hat_arrays = np.split(y_hat[1].detach().numpy(), future_f)
                    
                    miou = 0
                    for i in range(future_f):
                        label_seg_matrix = torch.tensor(get_segmentation_matrix(label_arrays[i]))
                        # print("label matrix", label_seg_matrix)
                        yhat_seg_matrix = torch.tensor(get_segmentation_matrix(y_hat_arrays[i]))
                        # print("yhat matrix", yhat_seg_matrix)
                        get_seg_viz(label_arrays[i], y_hat_arrays[i])

                        iou = calculate_iou(yhat_seg_matrix, label_seg_matrix)
                        # print("iou", iou)

                        miou += iou

                    miou /= future_f
                    # print("miou", miou)

                    train_miou += miou

                loss = criterion(y_hat, labels)
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1

                if not (i + 1) % 400:
                    print(
                        f"Epoch [{epoch+1}/{num_epochs}] Current time:{datetime.now()}"
                    )

            train_loss /= len(train_loader)
            train_miou /= len(train_loader)
            print(f"train loss: {train_loss:.4f}, train miou: {train_miou:.4f}")

            save_checkpoint(epoch, model, optimizer, checkpoint_file)

            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                batch_val_miou = 0.0
                for i, (val_images, val_labels) in enumerate(validation_loader):
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    val_outputs = model(val_images)

                    if n_th_frame is False:
                        val_label_arrays = np.split(val_labels[1].detach().numpy(), future_f)
                        val_y_hat_arrays = np.split(val_outputs[1].detach().numpy(), future_f)
                        
                        val_miou = 0.0
                        for i in range(future_f):
                            val_label_seg_matrix = torch.tensor(get_segmentation_matrix(val_label_arrays[i])) 
                            # print("label matrix", label_seg_matrix)
                            val_out_seg_matrix = torch.tensor(get_segmentation_matrix(val_y_hat_arrays[i]))
                            # print("yhat matrix", yhat_seg_matrix)
                            get_seg_viz(val_label_arrays[i], val_y_hat_arrays[i])

                            val_iou = calculate_iou(val_out_seg_matrix, val_label_seg_matrix)
                            # print("iou", iou)

                            val_miou += val_iou

                        val_miou /= future_f
                        # print("val_miou", val_miou)

                        batch_val_miou += val_miou
                        # print("batch_val_miou", batch_val_miou)

                    val_loss += criterion(val_outputs, val_labels).item()

                    output_folder = "visualizations/validation/" + model_name
                    visualize(labels, y_hat, output_folder, n_th_frame, future_f)

                val_loss /= len(validation_loader)
                batch_val_miou /= len(validation_loader)
                print(f"val loss: {val_loss:.4f}, val miou: {val_miou:.4f}")

            print(
                f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}, Training IOU: {train_miou:.4f}, Validation IOU: {batch_val_miou:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_miou = val_miou
                consecutive_no_improvement = 0
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    "model/" + model_name + "/best_model_checkpoint.pth",
                )
            else:
                consecutive_no_improvement += 1

            if consecutive_no_improvement >= patience:
                print(f"best_val_loss {best_val_loss}, best_val_iou {best_val_miou}")
                print(f"Early stopping at epoch {epoch+1}")
                break
            print(f"best_val_loss {best_val_loss}, best_val_iou {best_val_miou}")

    if test_flag:
        model.eval()
        se = 0
        samples_count = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                y_hat = model(labels)
                print(y_hat.shape)
                loss = criterion(y_hat, labels)

                se += loss.item() * labels.size(0)
                samples_count += labels.size(0)

                output_folder = "visualizations/test/" + model_name
                visualize(labels, y_hat, output_folder, n_th_frame, future_f)

        mse = se / samples_count
        print(f"MSE of test data: {mse:.3f}")


def save_checkpoint(epoch, model, optimizer, filename):
    print("Saving model checkpoint...")
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def visualize(viz_labels, viz_outputs, output_folder, n_th_frame, future_f):
    labels = viz_labels.view(viz_labels.size(0), 5, 100)
    y_hat = viz_outputs.view(viz_outputs.size(0), 5, 100)

    if n_th_frame:
        outer_loop = 1
        inner_loop = 1
    else:
        outer_loop = labels.size(1)
        inner_loop = future_f

    for i in range(outer_loop):
        label_frame = labels[i]
        y_hat_frame = y_hat[i]

        sample_folder = os.path.join(output_folder, f"sample_{i}")
        os.makedirs(sample_folder, exist_ok=True)

        for j in range(inner_loop):
            label_array = label_frame.cpu().detach().numpy()
            y_hat_array = y_hat_frame.cpu().detach().numpy()

            plt.figure(figsize=(8, 4))
            plt.plot(label_array[j], label="Label Array")
            plt.plot(y_hat_array[j], label="y_hat Array")
            plt.title(f"Frame {j}")
            plt.legend()

            plt.savefig(os.path.join(sample_folder, f"sample_{i}_frame_{j}.png"))
            plt.close()

def get_segmentation_matrix(y_array):
    # print("y_array", y_array)
    x_coordinates = np.arange(0, 100)

    # Plot the curve and segmentation matrix
    plt.plot(x_coordinates, y_array, label='Curve')
    plt.fill_between(x_coordinates, 0, y_array, alpha=0.4)
    plt.ylim(0, 300)
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.legend()
    plt.savefig(f"curve_segmentation.png")
    plt.close()

    img = Image.open("curve_segmentation.png").convert('L')
    segmentation_matrix = np.array(img)

    # Threshold the image to create a binary segmentation matrix

    # threshold = filters.threshold_otsu(segmentation_matrix)
    threshold = 255  # 255 for each R, G, B channel

    # Create a binary segmentation matrix
    binary_segmentation_matrix = (segmentation_matrix < threshold).astype(int)

    np.set_printoptions(threshold=np.inf)
    # print(binary_segmentation_matrix)

    return binary_segmentation_matrix


def calculate_iou(y_pred, y_true):
    intersection = torch.logical_and(y_pred, y_true).sum()
    union = torch.logical_or(y_pred, y_true).sum()
    iou = intersection.float() / union.float()
    return iou.item() 

def get_seg_viz(label_array, pred_array):
    # Plot the curve and segmentation matrix
    x_coordinates = np.arange(0, 100)
    plt.plot(x_coordinates, label_array, label='Curve')
    plt.plot(x_coordinates, pred_array, label='Curve')
    plt.fill_between(x_coordinates, 0, label_array, alpha=0.4)
    plt.fill_between(x_coordinates, 0, pred_array, alpha=0.4)
    plt.ylim(0, 300)
    plt.xlabel('X coordinates')
    plt.ylabel('Y coordinates')
    plt.legend()
    plt.savefig(f"curve_segmentation1.png")
    plt.close()