import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk
import threading
import numpy as np
import qrcode
import ffmpeg
from reedsolo import RSCodec, ReedSolomonError
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ProcessPoolExecutor, as_completed
import soundfile as sf
import multiprocessing
from colorsys import hls_to_rgb as hls_to_rgb_builtin

def hls_to_rgb(h, l, s):
    r, g, b = hls_to_rgb_builtin(h, l, s)
    return int(r * 255), int(g * 255), int(b * 255)

# Constants and defaults
DEFAULTS = {
    'pixel_block_width': 3,
    'pixel_block_height': 3,
    'video_width': 1920,
    'video_height': 1080,
    'fps': 30,
    'error_correction_percentage': 10,
    'gpu_index': 0,
    'use_hdr': False,
    'use_error_correction': True,
    'color_space': 'bt709',
    'font_size': 20,
    'num_threads': 4,
    'use_memory_cache': False,
    'memory_cache_size': 4  # Default cache size in GB
}

COLOR_HLS_VALUES = [(i / 16, 0.5, 1) for i in range(16)]
COLORS = {
    'START': (255, 255, 0),
    'END': (255, 0, 255),
    'MAP': {f'{i:X}': tuple(int(c * 255) for c in hls_to_rgb(*COLOR_HLS_VALUES[i])) for i in range(16)}
}

def clear_directory(dir_path):
    """Clear the specified directory. If it doesn't exist, create it."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def run_log_viewer():
    """Run the log viewer in a separate process."""
    os.system('python log.py')

class VideoFileConverterApp:
    def __init__(self, root):
        self.root = root
        self.mode_var = tk.StringVar(value="pixel")
        self.use_hdr_var = tk.BooleanVar(value=DEFAULTS['use_hdr'])
        self.use_error_correction_var = tk.BooleanVar(value=DEFAULTS['use_error_correction'])
        self.force_gpu_var = tk.BooleanVar(value=False)
        self.color_space_var = tk.StringVar(value=DEFAULTS['color_space'])
        self.auto_adjust_var = tk.BooleanVar(value=True)
        self.use_memory_cache_var = tk.BooleanVar(value=DEFAULTS['use_memory_cache'])
        self.memory_cache_size_var = tk.StringVar(value=str(DEFAULTS['memory_cache_size']))
        self.setup_ui()
        self.root.title("文件转视频工具，支持里德-所罗门纠错")
        clear_directory('cache')
        clear_directory('string')
        self.stop_saving_thread = False
        threading.Thread(target=run_log_viewer, daemon=True).start()

    def setup_ui(self):
        """Setup the UI."""
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text="将文件转换为带有里德-所罗门纠错的视频:").pack(pady=10)
        
        # Mode selection buttons
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(pady=5)
        ttk.Radiobutton(mode_frame, text="像素模式", variable=self.mode_var, value="pixel", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="文本模式", variable=self.mode_var, value="text", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="二维码模式", variable=self.mode_var, value="qrcode", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="音频模式", variable=self.mode_var, value="audio", command=self.update_ui).pack(side=tk.LEFT)
        
        # Configuration parameters input
        self.config_frame = ttk.Frame(frame)
        self.config_frame.pack(fill='both', expand=True)
        self.update_ui()
        
        # Start conversion button
        ttk.Button(frame, text="选择文件并转换为视频", command=self.convert_file_to_video).pack(pady=5)

    def update_ui(self):
        """Update the UI to show/hide different mode inputs."""
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        for name in DEFAULTS:
            if name not in ['use_hdr', 'use_error_correction', 'color_space', 'font_size', 'num_threads', 'use_memory_cache', 'memory_cache_size']:
                if self.mode_var.get() in ["text", "qrcode", "audio"] and name in ['pixel_block_width', 'pixel_block_height']:
                    continue
                ttk.Label(self.config_frame, text=f"{name.replace('_', ' ').title()}:").pack()
                var = tk.StringVar(value=str(DEFAULTS[name]))
                setattr(self, name, var)
                ttk.Entry(self.config_frame, textvariable=var).pack()
        
        # Specific configuration for modes
        if self.mode_var.get() == "qrcode":
            ttk.Checkbutton(self.config_frame, text="自动调整二维码大小", variable=self.auto_adjust_var, command=self.toggle_qr_options).pack(pady=5)
            self.qr_size_label = ttk.Label(self.config_frame, text="二维码大小:")
            self.qr_size_label.pack(pady=5)
            self.qr_size = tk.StringVar(value="10")
            self.qr_size_entry = ttk.Entry(self.config_frame, textvariable=self.qr_size)
            self.qr_size_entry.pack(pady=5)
            self.qr_number_label = ttk.Label(self.config_frame, text="二维码数量:")
            self.qr_number_label.pack(pady=5)
            self.qr_number = tk.StringVar(value="1")
            self.qr_number_entry = ttk.Entry(self.config_frame, textvariable=self.qr_number)
            self.qr_number_entry.pack(pady=5)
            self.toggle_qr_options()

        ttk.Checkbutton(self.config_frame, text="启用HDR", variable=self.use_hdr_var).pack(pady=5)
        ttk.Checkbutton(self.config_frame, text="启用纠错", variable=self.use_error_correction_var).pack(pady=5)
        ttk.Checkbutton(self.config_frame, text="强制使用GPU", variable=self.force_gpu_var).pack(pady=5)
        ttk.Checkbutton(self.config_frame, text="启用内存缓存", variable=self.use_memory_cache_var, command=self.toggle_memory_cache).pack(pady=5)
        self.memory_cache_size_label = ttk.Label(self.config_frame, text="内存缓存大小（GB）:")
        self.memory_cache_size_label.pack(pady=5)
        self.memory_cache_size_entry = ttk.Entry(self.config_frame, textvariable=self.memory_cache_size_var)
        self.memory_cache_size_entry.pack(pady=5)
        self.toggle_memory_cache()
        
        ttk.Label(self.config_frame, text="色彩空间:").pack()
        self.color_space_menu = ttk.Combobox(self.config_frame, textvariable=self.color_space_var, values=["bt709 (sRGB)", "bt2020 (HDR)"])
        self.color_space_menu.pack(pady=5)
        if self.mode_var.get() == "text":
            ttk.Label(self.config_frame, text="字体大小:").pack()
            self.font_size = tk.StringVar(value=str(DEFAULTS['font_size']))
            ttk.Entry(self.config_frame, textvariable=self.font_size).pack()
        if self.mode_var.get() == "audio":
            ttk.Label(self.config_frame, text="音频持续时间(秒):").pack()
            self.audio_duration = tk.StringVar(value="1.0")
            ttk.Entry(self.config_frame, textvariable=self.audio_duration).pack()
        ttk.Label(self.config_frame, text="线程数量:").pack()
        self.num_threads = tk.StringVar(value=str(DEFAULTS['num_threads']))
        ttk.Entry(self.config_frame, textvariable=self.num_threads).pack()

    def toggle_qr_options(self):
        """Show or hide QR code size and number options based on auto adjust selection."""
        if self.auto_adjust_var.get():
            self.qr_size_label.pack_forget()
            self.qr_size_entry.pack_forget()
            self.qr_number_label.pack_forget()
            self.qr_number_entry.pack_forget()
        else:
            self.qr_size_label.pack(pady=5)
            self.qr_size_entry.pack(pady=5)
            self.qr_number_label.pack(pady=5)
            self.qr_number_entry.pack(pady=5)

    def toggle_memory_cache(self):
        """Show or hide memory cache size option based on memory cache selection."""
        if self.use_memory_cache_var.get():
            self.memory_cache_size_label.pack(pady=5)
            self.memory_cache_size_entry.pack(pady=5)
        else:
            self.memory_cache_size_label.pack_forget()
            self.memory_cache_size_entry.pack_forget()

    def convert_file_to_video(self):
        """Select file and start conversion to video."""
        file_path = filedialog.askopenfilename()
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if not file_path or not output_path:
            return
        self.log_message(f"选择的文件: {file_path}")
        self.log_message(f"输出路径: {output_path}")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        threading.Thread(target=self.process_file_to_video, args=(file_path, output_path), daemon=True).start()

    def process_file_to_video(self, file_path, output_path):
        """Process the file and convert it to a video."""
        try:
            # Get and convert user input parameters
            self.video_width = int(self.video_width.get())
            self.video_height = int(self.video_height.get())
            self.pixel_block_width = int(self.pixel_block_width.get())
            self.pixel_block_height = int(self.pixel_block_height.get())
            self.fps = int(self.fps.get())
            self.error_correction_percentage = int(self.error_correction_percentage.get())
            self.gpu_index = int(self.gpu_index.get())
            self.font_size = int(self.font_size.get()) if self.mode_var.get() == "text" else None
            self.num_threads = int(self.num_threads.get())
            self.use_memory_cache = self.use_memory_cache_var.get()
            self.memory_cache_size = int(self.memory_cache_size_var.get()) * 1024 * 1024 * 1024  # Convert to bytes
            self.audio_duration = float(self.audio_duration.get()) if self.mode_var.get() == "audio" else None

            # Read file data
            data = self.read_file_data(file_path)
            if not data:
                return self.log_message("错误: 无法读取文件数据。")
            self.log_message(f"文件读取成功，大小: {len(data)} 字节。")
            
            # Data encoding
            if self.use_error_correction_var.get():
                rs_n, rs_k = 255, 255 - (255 * self.error_correction_percentage // 100)
                encoded_data = self.encode_data(data, rs_n, rs_k)
                if not encoded_data:
                    return self.log_message("错误: 数据编码失败。")
                self.log_message(f"数据编码成功，编码后大小: {len(encoded_data)} 字节。")
            else:
                encoded_data = data

            # Write encoded data to cache file
            with open("cache.txt", "w") as cache_file:
                if self.mode_var.get() == "audio":
                    cache_file.write(encoded_data.hex())
                else:
                    cache_file.write(data.hex())

            # Create video based on selected mode
            if self.mode_var.get() == "pixel":
                self.create_video_pixel_mode(encoded_data, output_path)
            elif self.mode_var.get() == "text":
                self.create_video_text_mode(encoded_data, output_path)
            elif self.mode_var.get() == "qrcode":
                self.create_video_qrcode_mode(encoded_data, output_path)
            elif self.mode_var.get() == "audio":
                self.create_audio_mode(encoded_data, output_path)
            self.log_message("处理完成。")
        except Exception as e:
            self.log_message(f"错误: {str(e)}")

    def read_file_data(self, file_path):
        """Read file data."""
        self.log_message("读取文件数据...")
        with open(file_path, 'rb') as file:
            data = file.read()
        self.log_message(f"文件读取成功，大小: {len(data)} 字节")
        return data

    def encode_data(self, data, rs_n, rs_k):
        """Encode data using Reed-Solomon encoding."""
        self.log_message(f"使用里德-所罗门编码 (rs_n={rs_n}, rs_k={rs_k}) 对数据进行编码...")
        rs = RSCodec(rs_n - rs_k)
        try:
            encoded_data = rs.encode(data)
            self.log_message(f"数据编码成功: {len(encoded_data)} 字节")
            return encoded_data
        except ReedSolomonError as e:
            self.log_message(f"里德-所罗门编码错误: {str(e)}")
            return None

    def create_video_pixel_mode(self, data, output_path):
        self.log_message("开始创建像素模式的视频...")
        frame_size = (self.video_width // self.pixel_block_width) * (self.video_height // self.pixel_block_height)
        num_frames = (len(data) + frame_size - 1) // frame_size
        padded_data = data + bytes([0] * (num_frames * frame_size - len(data)))
        total_frames = num_frames + 2  # 包含开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

        try:
            pix_fmt, vcodec = ('yuv420p10le', 'libx265') if self.use_hdr_var.get() else ('yuv420p', 'libx264')
            color_primaries, color_trc, colorspace = ('bt2020', 'smpte2084', 'bt2020nc') if self.color_space_var.get() == 'bt2020 (HDR)' else ('bt709', 'bt709', 'bt709')

            process = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.video_width}x{self.video_height}', framerate=self.fps)
                .output(output_path, pix_fmt=pix_fmt, vcodec=vcodec, preset='veryslow', crf=0, video_bitrate='2000k', maxrate='2000k', bufsize='4000k', color_primaries=color_primaries, color_trc=color_trc, colorspace=colorspace, rc_lookahead=32)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['START']).tobytes())
            self.log_message("写入开始帧...")

            with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {executor.submit(self.process_frame, padded_data[i * frame_size:(i + 1) * frame_size]): i for i in range(num_frames)}
                results = [None] * num_frames
                for future in as_completed(futures):
                    i, frame_data = futures[future], future.result()
                    results[i] = frame_data
                    self.log_message(f"处理帧 {i + 1}/{total_frames}。")

            for i, frame_data in enumerate(results):
                if frame_data is None:
                    self.log_message(f"错误: 帧 {i + 1} 数据为空")
                    continue
                process.stdin.write(frame_data)

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def create_video_text_mode(self, data, output_path):
        self.log_message("开始创建文本模式的视频...")
        hex_data = data.hex().upper()
        num_frames = (len(hex_data) + (self.video_width * self.video_height // (self.font_size * self.font_size)) - 1) // (self.video_width * self.video_height // (self.font_size * self.font_size))
        padded_data = hex_data.ljust(num_frames * self.video_width * self.video_height // (self.font_size * self.font_size), '0')
        total_frames = num_frames + 2  # 包括开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

        try:
            pix_fmt, vcodec = ('yuv420p10le', 'libx265') if self.use_hdr_var.get() else ('yuv420p', 'libx264')
            color_primaries, color_trc, colorspace = ('bt2020', 'smpte2084', 'bt2020nc') if self.color_space_var.get() == 'bt2020 (HDR)' else ('bt709', 'bt709', 'bt709')

            process = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.video_width}x{self.video_height}', framerate=self.fps)
                .output(output_path, pix_fmt=pix_fmt, vcodec=vcodec, preset='veryslow', crf=0, video_bitrate='2000k', maxrate='2000k', bufsize='4000k', color_primaries=color_primaries, color_trc=color_trc, colorspace=colorspace, rc_lookahead=32)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['START']).tobytes())
            self.log_message("写入开始帧...")

            self.generate_text_images(self.font_size)

            with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {executor.submit(self.process_text_frame, padded_data[i * (self.video_width * self.video_height // (self.font_size * self.font_size)):(i + 1) * (self.video_width * self.video_height // (self.font_size * self.font_size))], self.font_size): i for i in range(num_frames)}
                results = [None] * num_frames
                for future in as_completed(futures):
                    i, frame_data = futures[future], future.result()
                    results[i] = frame_data
                    self.log_message(f"处理帧 {i + 1}/{total_frames}。")

            for i, frame_data in enumerate(results):
                if frame_data is None:
                    self.log_message(f"错误: 帧 {i + 1} 数据为空")
                    continue
                process.stdin.write(frame_data)

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def create_video_qrcode_mode(self, data, output_path):
        self.log_message("开始创建二维码模式的视频...")
        chunk_size = 1024
        num_frames = (len(data) + chunk_size - 1) // chunk_size
        padded_data = data + bytes([0] * (num_frames * chunk_size - len(data)))
        total_frames = num_frames + 2  # 包括开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

        try:
            pix_fmt, vcodec = ('yuv420p10le', 'libx265') if self.use_hdr_var.get() else ('yuv420p', 'libx264')
            color_primaries, color_trc, colorspace = ('bt2020', 'smpte2084', 'bt2020nc') if self.color_space_var.get() == 'bt2020 (HDR)' else ('bt709', 'bt709', 'bt709')

            process = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.video_width}x{self.video_height}', framerate=self.fps)
                .output(output_path, pix_fmt=pix_fmt, vcodec=vcodec, preset='veryslow', crf=0, video_bitrate='2000k', maxrate='2000k', bufsize='4000k', color_primaries=color_primaries, color_trc=color_trc, colorspace=colorspace, rc_lookahead=32)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['START']).tobytes())
            self.log_message("写入开始帧...")

            with ProcessPoolExecutor(max_workers=self.num_threads) as executor:
                futures = {executor.submit(self.process_qrcode_frame, padded_data[i * chunk_size:(i + 1) * chunk_size]): i for i in range(num_frames)}
                results = [None] * num_frames
                for future in as_completed(futures):
                    i, frame_data = futures[future], future.result()
                    results[i] = frame_data
                    self.log_message(f"处理帧 {i + 1}/{total_frames}。")

            for i, frame_data in enumerate(results):
                if frame_data is None:
                    self.log_message(f"错误: 帧 {i + 1} 数据为空")
                    continue
                process.stdin.write(frame_data)

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def create_audio_mode(self, data, output_path):
        self.log_message("开始创建音频模式...")
        data_length = len(data) * 8  # Convert bytes to bits
        num_frequencies = 20000  # Number of frequencies (20000 Hz)
        duration = self.audio_duration
        total_frames = (data_length + num_frequencies - 1) // num_frequencies
        padded_data = data + bytes([0] * ((total_frames * num_frequencies // 8) - len(data)))
        self.log_message(f"音频将包含 {total_frames} 帧（包括开始和结束帧）。")

        try:
            audio_data = np.zeros((total_frames * int(44100 * duration),), dtype=np.float32)

            for frame in range(total_frames):
                self.log_message(f"处理音频帧 {frame + 1}/{total_frames}。")
                start_bit = frame * num_frequencies
                for i in range(num_frequencies):
                    bit_index = start_bit + i
                    byte_index = bit_index // 8
                    bit_offset = 7 - (bit_index % 8)
                    bit_value = (padded_data[byte_index] >> bit_offset) & 1
                    if bit_value:
                        t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
                        frequency = i + 1
                        audio_data[frame * int(44100 * duration):(frame + 1) * int(44100 * duration)] += np.sin(2 * np.pi * frequency * t)

            # Normalize audio data
            audio_data /= np.max(np.abs(audio_data))

            # Write audio data to file
            sf.write(output_path, audio_data, 44100, 'PCM_16')
            self.log_message("音频处理完成。")
        except Exception as e:
            self.log_message(f"音频创建过程中出错: {str(e)}")

    def create_solid_color_frame(self, width, height, color):
        """Create a solid color frame."""
        return np.full((height, width, 3), color, dtype=np.uint8)

    def process_frame(self, frame_data):
        """Process frame data for pixel mode."""
        image = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        blocks_per_row = self.video_width // self.pixel_block_width
        for i in range(len(frame_data)):
            block_x = (i % blocks_per_row) * self.pixel_block_width
            block_y = (i // blocks_per_row) * self.pixel_block_height
            color = COLORS['MAP']['{:X}'.format(frame_data[i] % 16)]
            image[block_y:block_y + self.pixel_block_height, block_x:block_x + self.pixel_block_width] = color
        return image.tobytes()

    def process_text_frame(self, frame_data, font_size):
        """Process frame data for text mode."""
        image = Image.new('RGB', (self.video_width, self.video_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        x, y = 0, 0
        for char in frame_data:
            draw.text((x, y), char, font=font, fill=(0, 0, 0))
            x += font_size
            if x >= self.video_width:
                x = 0
                y += font_size
        return np.array(image).tobytes()

    def process_qrcode_frame(self, frame_data):
        """Process frame data for QR code mode."""
        qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=1)
        qr.add_data(frame_data)
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert('RGB')
        qr_image = qr_image.resize((self.video_width, self.video_height), Image.Resampling.LANCZOS)
        return np.array(qr_image).tobytes()

    def log_message(self, message):
        """Log a message to the log viewer."""
        with open('log.txt', 'a') as log_file:
            log_file.write(message + '\n')

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFileConverterApp(root)
    root.mainloop()
