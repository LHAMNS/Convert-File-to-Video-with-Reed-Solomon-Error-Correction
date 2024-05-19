
# File to Video Converter with Reed-Solomon Error Correction

This project converts any file into a video using Reed-Solomon error correction, allowing for secure and resilient storage or transmission of the data. The video can be encoded in two modes: Pixel Mode and Text Mode. The output video can also utilize HDR for better color representation.

## Features

- Convert any file to a video with Reed-Solomon error correction.
- Two encoding modes:
  - **Pixel Mode:** Uses colored pixels to represent the data.
  - **Text Mode:** Uses hexadecimal text to represent the data.
- Optional HDR encoding (HDR10).
- Supports GPU acceleration using CUDA.
- Adjustable parameters for video resolution, frame rate, and error correction percentage.
- Detailed progress logging.

## Requirements

- Python 3.6+
- ffmpeg
- NumPy
- Pillow
- reedsolo
- Tkinter
- CUDA and CuPy (optional for GPU acceleration)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install numpy pillow reedsolo tk
   ```

2. **Install ffmpeg:**
   Follow the instructions on [ffmpeg.org](https://ffmpeg.org/download.html) to download and install ffmpeg for your operating system.

3. **Install CUDA and CuPy (optional for GPU acceleration):**
   - Follow the instructions on [NVIDIA's CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) for your operating system.
   - Install CuPy:
     ```bash
     pip install cupy-cudaXXX  # Replace XXX with your CUDA version, e.g., cupy-cuda112 for CUDA 11.2
     ```

## Usage

1. **Run the script:**
   ```bash
   python path/to/your/script.py
   ```

2. **Configure the parameters:**
   - Select Pixel Mode or Text Mode.
   - Set the video width, height, and frame rate.
   - Set the error correction percentage (optional).
   - Set the GPU index (optional).
   - Set the number of threads for processing.
   - Enable HDR (optional).
   - Enable error correction (optional).
   - Choose the color space (bt709 or bt2020).
   - Set the font size for Text Mode.

3. **Convert a file to video:**
   - Click the "Select File and Convert to Video" button.
   - Choose the input file and specify the output file path.

4. **Monitor the progress:**
   - Detailed log messages will appear in the status window.
   - The progress bar will indicate the processing status.

## Example

1. **Select the mode:**
   ![Mode Selection](images/mode_selection.png)

2. **Configure parameters:**
   ![Parameter Configuration](images/parameter_configuration.png)

3. **Conversion progress:**
   ![Conversion Progress](images/conversion_progress.png)

## Notes

- Ensure you have enough memory and disk space for large files.
- If you encounter any issues, please check the log messages for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
