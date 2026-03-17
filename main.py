import os
import pims
import numpy as np
import logging
import tifffile
from functools import wraps
from pathlib import Path
from tiff_aoi_extractor import read_tiff_stack
from gaussian_spot_model import generate_gaussian_spot
from snr_calc import compute_snr_for_aois
from analyze_config import analyze_configuration_from_path_string
from plotting_blue_channel import plot_blue_bypass_vs_split
from plotting_red_channel import plot_red_bypass_vs_split
from assess_crosstalk import assess_crosstalk 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_input_subfolder(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        os.makedirs(self.input_subfolder, exist_ok=True)
        return func(self, *args, **kwargs)
    return wrapper

class SPEBatchProcessor:
    def __init__(self, input_root):
        self.input_root = input_root

    def process_all_spe_files(self):
        self.averaged_images = []  # Collect averaged images
        for root, _, files in os.walk(self.input_root):
            for file in files:
                if file.endswith('.spe') or file.endswith('.SPE'):
                    spe_file_path = os.path.join(root, file)
                    logging.info(f"Processing {spe_file_path}: averaging frames")
                    
                    # Read and average frames in memory
                    frames = pims.open(spe_file_path)
                    frame_list = []
                    for frame in frames:
                        if frame.dtype != np.uint16:
                            frame = (frame / frame.max() * 65535).astype(np.uint16)
                        frame_list.append(frame.astype(np.float32))
                    
                    if frame_list:
                        avg_img = np.mean(np.stack(frame_list, axis=0), axis=0)
                        self.averaged_images.append(avg_img)
                        logging.info(f"Averaged {len(frame_list)} frames from {file}")
                    else:
                        logging.warning(f"No frames in {file}")

    def average_frames_to_zstack(self, output_tiff_path):
        if not self.averaged_images:
            logging.error("No averaged images.")
            return

        # Stack into z-stack
        z_stack = np.stack(self.averaged_images, axis=0)
        tifffile.imwrite(output_tiff_path, z_stack.astype(np.uint16))
        logging.info(f"Averaged z-stack saved to {output_tiff_path}")       

def process_spe_files():
    while True:
        input_root = input('Please enter the path with your files in the .spe format: ')
        if os.path.isdir(input_root):
            break
        print("The specified path does not exist. Please try again.")

    batch_processor = SPEBatchProcessor(input_root)
    batch_processor.process_all_spe_files()

    output_tiff = input('Enter output path for averaged_zstack.tiff: ')
    if os.path.isdir(output_tiff):
        output_tiff = os.path.join(output_tiff, "averaged_zstack.tiff")
    batch_processor.average_frames_to_zstack(output_tiff)

    print(f"Averaged z-stack saved to {output_tiff}.")

def extract_aois():
    while True:
        output_path = input("Please provide the output directory for AOI extraction: ")
        if os.path.isdir(output_path):
            break
        print("The specified path does not exist. Please try again.")

    tiff_path = input("Please provide the path to the TIFF z-stack file or 2D gaussian spot image: ")
    if not os.path.isfile(tiff_path):
        print("TIFF file does not exist.")
        return

    positions_input = input("Please provide AOI positions as comma-separated x,y pairs (e.g., 100,200;150,250): ")
    try:
        positions = [tuple(map(int, pos.split(','))) for pos in positions_input.split(';')]
    except ValueError:
        print("Invalid positions format.")
        return

    offset_x = int(input("Offset X coordinate: "))
    offset_y = int(input("Offset Y coordinate: "))
    offset_P = int(input("Offset size (P): "))

    # Simple progress bar function
    def progress_bar(iterable):
        for i, item in enumerate(iterable):
            print(f"Processing {i+1}/{len(iterable)}", end='\r')
            yield item
        print()

    kwargs = {
        "tiff_path": tiff_path,
        "positions": positions,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "offset_P": offset_P,
    }

    read_tiff_stack(Path(output_path), progress_bar, **kwargs)

    # Save the manual centroid positions entered by the user
    coords_path = Path(output_path) / "positions.npy"
    np.save(coords_path, np.array(positions))
    print(f"Saved manual AOI coordinates to {coords_path}")

def model_gaussians():
    output_path = input("Please provide the output directory to save the Gaussian spot: ")
    if not os.path.isdir(output_path):
        print("Directory does not exist.")
        return

    try:
        height = float(input("Spot intensity (height): "))
        width = float(input("Spot width: "))
        x = float(input("Spot center x relative to target: "))
        y = float(input("Spot center y relative to target: "))
        target_x = float(input("Target x position: "))
        target_y = float(input("Target y position: "))
        P = int(input("Image size (P, e.g., 50): "))
    except ValueError:
        print("Invalid input values.")
        return

    spot_image = generate_gaussian_spot(height, width, x, y, target_x, target_y, P)
    output_file = os.path.join(output_path, "gaussian_spot.tiff")
    tifffile.imwrite(output_file, spot_image.astype(np.float32))
    print(f"Gaussian spot saved to {output_file}")

def calculate_aoi_snr():
    while True:
        data_path = input("Please provide the path to the AOI extraction output directory (containing data.npy, etc.): ")
        if os.path.isdir(data_path):
            break
        print("The specified path does not exist. Please try again.")

    data_path = Path(data_path)
    if not (data_path / "data.npy").exists():
        print("data.npy not found in the directory.")
        return

    # Load offset data
    offset_samples = np.load(data_path / "offset_samples.npy")
    offset_weights = np.load(data_path / "offset_weights.npy")

    compute_snr_for_aois(data_path, offset_samples, offset_weights)
    
def aggregate_config_snr():
    """
    Ask user for configuration results root and call analyze_config.
    """
    while True:
        results_root = input(
            "Please provide the configuration results directory "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_bypass_blue\\results): "
        ).strip()
        if os.path.isdir(results_root):
            break
        print("The specified path does not exist. Please try again.")

    # New prompt for channel tagging
    channel_choice = input("Enter channel for these results (blue/red/none): ").strip().lower()
    channel = channel_choice if channel_choice in ["blue", "red"] else None

    analyze_configuration_from_path_string(results_root, channel)

def plot_blue_bypass_vs_split_menu():
    while True:
        bypass_root = input(
            "Path to BLUE bypass results folder "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_bypass_blue\\results): "
        ).strip()
        if os.path.isdir(bypass_root):
            break
        print("The specified path does not exist. Please try again.")

    while True:
        split_root = input(
            "Path to BLUE split results folder "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_split_blue\\results): "
        ).strip()
        if os.path.isdir(split_root):
            break
        print("The specified path does not exist. Please try again.")

    print("Generating plots for blue bypass vs blue split...")
    plot_blue_bypass_vs_split(bypass_root, split_root)

def plot_red_bypass_vs_split_menu():
    while True:
        bypass_root = input(
            "Path to RED bypass results folder "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_bypass_red\\results): "
        ).strip()
        if os.path.isdir(bypass_root):
            break
        print("The specified path does not exist. Please try again.")

    while True:
        split_root = input(
            "Path to RED split results folder "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_split_red\\results): "
        ).strip()
        if os.path.isdir(split_root):
            break
        print("The specified path does not exist. Please try again.")

    print("Generating plots for red bypass vs red split...")
    plot_red_bypass_vs_split(bypass_root, split_root)

def assess_crosstalk_menu():
    """
    Menu wrapper for crosstalk assessment: prompt for results root path.
    """
    while True:
        results_root = input(
            "Please provide the dual-channel results directory "
            "(e.g. D:\\tu_projects\\benchmarking\\data\\beads_dual_red_blue\\results): "
        ).strip()
        if os.path.isdir(results_root):
            break
        print("The specified path does not exist. Please try again.")

    print("Assessing crosstalk...")
    assess_crosstalk(results_root)

def main():
    while True:
        try:
            print("\nChoose an option:")
            print("1. Process .spe files and create averaged z-stack in .tiff format")
            print("2. Extract AOIs from TIFF z-stack")
            print("3. Model spots as Gaussians")
            print("4. Calculate SNR for AOIs")
            print("5. Aggregate SNR over configuration (pos1, pos2, ...)")
            print("6. Plotting: blue bypass vs blue split")
            print("7. Plotting: red bypass vs red split")
            print("8. Assess crosstalk")
            print("9. Exit")
            choice = input("Enter your choice (1-9): ").strip()

            if choice == '1':
                print("Processing .spe files and creating averaged z-stack in .tiff format...")
                process_spe_files()
            elif choice == '2':
                print("Extracting AOIs from TIFF z-stack...")
                extract_aois()
            elif choice == '3':
                print("Modeling spots as Gaussians...")
                model_gaussians()
            elif choice == '4':
                print("Calculating SNR for AOIs...")
                calculate_aoi_snr()
            elif choice == '5':
                print("Aggregating SNR over configuration...")
                aggregate_config_snr()
            elif choice == '6':
                print("Plotting: blue bypass vs blue split...")
                plot_blue_bypass_vs_split_menu()
            elif choice == '7':
                print("Plotting: red bypass vs red split...")
                plot_red_bypass_vs_split_menu()
            elif choice == '8':
                print("Assessing Crosstalk...")
                assess_crosstalk_menu()
            elif choice == '9':
                print("Exiting...") 
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Exiting...")
            break
        except Exception as e:
            import traceback
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            break
if __name__ == "__main__":
    main()