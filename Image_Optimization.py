import os
import sys
import numpy as np
import cv2
import torch
from torch import Tensor
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import pandas as pd
import glob
import argparse
import subprocess
import random

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


def make_unique_folder(base_folder):
    """
    Create a unique folder with incremental numbering if the base folder exists.
    """
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        return base_folder
    else:
        counter = 1
        while True:
            new_folder = f"{base_folder}_{counter}"
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
                return new_folder
            counter += 1


def run_yolo_detection(source, yolo_results_folder):
    """
    Run YOLOv5 detection and save results in the specified folder.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_script = os.path.join(current_dir, "yolov5", "detect.py")
    weights = os.path.join(current_dir, "yolov5", "weights", "best.pt")
    imgsz = 1280
    max_det = 1
    exp_name = "exp"  # Default YOLO experiment name
    coord_folder = os.path.join(yolo_results_folder, exp_name, "labels")

    if os.path.exists(coord_folder) and os.listdir(coord_folder):
        print(f"YOLO results already exist in: {coord_folder}. Skipping detection.")
        return coord_folder

    python_executable = sys.executable
    command = [
        python_executable, yolo_script,
        "--weights", weights,
        "--source", source,
        "--save-txt",
        "--imgsz", str(imgsz),
        "--max-det", str(max_det),
        "--project", yolo_results_folder,
        "--name", exp_name
    ]

    print(f"Running YOLOv5 command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"YOLOv5 detection failed with error: {e}")
        raise e

    return coord_folder



def crop_from_original(image_path, coords):
    """
    Crops regions from the original image based on YOLO coordinates.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    h, w = image.shape
    cropped_images = []

    for coord in coords:
        values = coord.split()
        if len(values) != 5:
            print(f"Warning: Invalid YOLO coordinate format: {coord}. Skipping.")
            continue

        _, x_center, y_center, bbox_width, bbox_height = map(float, values)
        x_center, y_center = x_center * w, y_center * h
        bbox_width, bbox_height = bbox_width * w, bbox_height * h

        x_min = max(0, int(x_center - bbox_width / 2))
        y_min = max(0, int(y_center - bbox_height / 2))
        x_max = min(w, int(x_center + bbox_width / 2))
        y_max = min(h, int(y_center + bbox_height / 2))

        cropped = image[y_min:y_max, x_min:x_max]
        cropped_images.append(torch.tensor(cropped, dtype=torch.float32))

    return cropped_images


def load_yolo_results_and_crop(source_folder, coord_folder, crop_save_folder):
    """
    Maps YOLO results to original 16-bit images and crops regions.
    """
    cropped_data = []

    if not os.path.exists(crop_save_folder):
        os.makedirs(crop_save_folder)

    coord_files = [f for f in os.listdir(coord_folder) if f.endswith(".txt")]

    for coord_file in tqdm(coord_files, desc="Processing YOLO coordinate files"):
        image_name = os.path.splitext(coord_file)[0]

        matching_images = glob.glob(os.path.join(source_folder, f"{image_name}.*"))
        if not matching_images:
            print(f"Warning: No matching image found for {image_name}. Skipping.")
            continue

        image_path = matching_images[0]

        with open(os.path.join(coord_folder, coord_file), 'r') as f:
            coords = [line.strip() for line in f.readlines()]

        cropped_images = crop_from_original(image_path, coords)

        for i, cropped_image in enumerate(tqdm(cropped_images, desc=f"Cropping {image_name}", leave=False)):
            crop_path = os.path.join(crop_save_folder, f"{image_name}_{i}.png")
            cropped_image_np = cropped_image.cpu().numpy().astype(np.uint16)
            cv2.imwrite(crop_path, cropped_image_np)

            cropped_data.append({
                "image_name": image_name,
                "cropped_index": i,
                "cropped_image": cropped_image
            })

    return cropped_data


def apply_histogram_equalization_torch(image: Tensor) -> Tensor:
    hist = torch.histc(image, bins=65536, min=0, max=65535)
    cdf = hist.cumsum(0)
    cdf_normalized = cdf / cdf[-1] * 65535
    equalized = cdf_normalized[image.long()]
    return equalized.view(image.size())


def adjust_brightness_contrast_torch(image: Tensor, alpha: float, beta: float) -> Tensor:
    adjusted = alpha * image + beta
    adjusted = torch.clamp(adjusted, 0, 65535)
    return adjusted


def apply_preprocessing_torch(image: Tensor, alpha: float, beta: float):
    processed_image = adjust_brightness_contrast_torch(image, alpha, beta)
    processed_image = apply_histogram_equalization_torch(processed_image)
    return processed_image


def objective_function_gpu(individual, image: Tensor):
    alpha, beta = individual[0], individual[1]
    processed_image = apply_preprocessing_torch(image, alpha, beta)
    return processed_image.var().item(),


def optimize_image_gpu(image: Tensor, ngen, pop_size, brightness_range, contrast_range, patience, delta):
    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_alpha", np.random.uniform, contrast_range[0], contrast_range[1])
    toolbox.register("attr_beta", np.random.uniform, brightness_range[0], brightness_range[1])
    toolbox.register("individual", tools.initCycle, creator.Individual, 
                     (toolbox.attr_alpha, toolbox.attr_beta), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective_function_gpu, image=image)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    best_score = -1
    no_improve_count = 0
    early_stop_gen = None
    best_individual = None

    for gen in tqdm(range(ngen), desc="Optimizing Image"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = list(toolbox.map(toolbox.evaluate, offspring))

        generation_best_score = max(fit[0] for fit in fits)
        if generation_best_score > best_score + delta:
            best_score = generation_best_score
            no_improve_count = 0
            best_individual = next(ind for ind, fit in zip(offspring, fits) if fit[0] == best_score)
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            early_stop_gen = gen + 1
            break

        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

    alpha, beta = best_individual[0], best_individual[1]
    return alpha, beta, best_score, early_stop_gen


def process_full_images(csv_path, source_folder, save_folder):
    """
    Applies preprocessing to full original images based on optimization results in CSV.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    results_df = pd.read_csv(csv_path)

    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Processing Full Images"):
        image_name = row["Image Name"]
        alpha = row["Alpha (Contrast)"]
        beta = row["Beta (Brightness)"]

        matching_images = glob.glob(os.path.join(source_folder, f"{os.path.splitext(image_name)[0]}.*"))
        if not matching_images:
            print(f"Warning: No matching original image found for {image_name}. Skipping.")
            continue

        original_image_path = matching_images[0]
        original_image = cv2.imread(original_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        original_tensor = torch.tensor(original_image, dtype=torch.float32)

        processed_tensor = apply_preprocessing_torch(original_tensor, alpha, beta)
        processed_image = processed_tensor.cpu().numpy().astype(np.uint16)

        save_path = os.path.join(save_folder, f"{os.path.splitext(image_name)[0]}_processed.png")
        cv2.imwrite(save_path, processed_image)


def process_yolo_results(source, result_folder, ngen, pop_size, brightness_range, contrast_range, patience, delta, use_gpu):
    # Define folders for results
    yolo_results_folder = os.path.join(result_folder, "yolo_results")
    crop_save_folder = os.path.join(result_folder, "crops")
    optimized_save_folder = os.path.join(result_folder, "optimized")
    full_processed_folder = os.path.join(result_folder, "full_processed")

    if not os.path.exists(optimized_save_folder):
        os.makedirs(optimized_save_folder)

    # Run YOLO detection and get coordinates
    coord_folder = run_yolo_detection(source, yolo_results_folder)

    # Crop images based on YOLO results
    cropped_data = load_yolo_results_and_crop(source, coord_folder, crop_save_folder)

    # Perform GA optimization on cropped images
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    results = []
    for data in tqdm(cropped_data, desc="Optimizing Cropped Images"):
        image_tensor = data["cropped_image"].to(device)

        alpha, beta, _, _ = optimize_image_gpu(
            image_tensor, ngen, pop_size, brightness_range, contrast_range, patience, delta
        )

        optimized_image = apply_preprocessing_torch(image_tensor, alpha, beta)
        optimized_image_np = optimized_image.cpu().numpy().astype(np.uint16)
        optimized_save_path = os.path.join(optimized_save_folder, f"{data['image_name']}_{data['cropped_index']}.png")
        cv2.imwrite(optimized_save_path, optimized_image_np)

        results.append({
            "Image Name": data["image_name"],
            "Alpha (Contrast)": alpha,
            "Beta (Brightness)": beta
        })

    # Save optimization results to CSV
    csv_path = os.path.join(result_folder, "optimization_results.csv")
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)

    # Apply optimization results to full original images
    process_full_images(csv_path, source, full_processed_folder)


def main():
    parser = argparse.ArgumentParser(description="Image Processing and Optimization using YOLOv5")
    parser.add_argument("--source", type=str, required=True, help="Path to the source image folder")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for processing (default: False)")
    parser.add_argument("--ngen", type=int, default=50, help="Number of generations for optimization (default: 50)")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size for optimization (default: 20)")
    parser.add_argument("--brightness_range", type=float, nargs=2, default=(-500, 500), help="Brightness range (default: -500 to 500)")
    parser.add_argument("--contrast_range", type=float, nargs=2, default=(0.5, 1.5), help="Contrast range (default: 0.5 to 1.5)")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping (default: 10)")
    parser.add_argument("--delta", type=float, default=1e-4, help="Delta for improvement threshold (default: 1e-4)")

    args = parser.parse_args()

    result_folder = make_unique_folder("./result_images")

    process_yolo_results(
        args.source,
        result_folder,
        args.ngen,
        args.pop_size,
        args.brightness_range,
        args.contrast_range,
        args.patience,
        args.delta,
        args.use_gpu
    )


if __name__ == "__main__":
    main()