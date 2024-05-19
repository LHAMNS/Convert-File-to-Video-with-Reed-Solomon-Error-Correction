import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import numpy as np
import ffmpeg, threading, os
from reedsolo import RSCodec, ReedSolomonError
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorsys import hls_to_rgb

try:
    import cupy as cp
    CUDA_ENABLED = True
except ImportError:
    CUDA_ENABLED = False

DEFAULTS = {'pixel_block_width': 3, 'pixel_block_height': 3, 'video_width': 1920, 'video_height': 1080, 'fps': 30, 'error_correction_percentage': 10, 'gpu_index': 0, 'use_hdr': False, 'use_error_correction': True, 'color_space': 'bt709', 'font_size': 20, 'num_threads': 4}

COLOR_HLS_VALUES = [(0, 0.5, 1), (30/360, 0.5, 1), (60/360, 0.5, 1), (90/360, 0.5, 1), (120/360, 0.5, 1), (150/360, 0.5, 1), (180/360, 0.5, 1), (210/360, 0.5, 1), (240/360, 0.5, 1), (270/360, 0.5, 1), (300/360, 0.5, 1), (330/360, 0.5, 1), (0, 0.25, 1), (0, 0.75, 1), (120/360, 0.25, 1), (120/360, 0.75, 1)]

def hls_to_rgb_tuple(h, l, s): return tuple(int(c * 255) for c in hls_to_rgb(h, l, s))

COLORS = {'START': (255, 255, 0), 'END': (255, 0, 255), 'MAP': {f'{i:X}': hls_to_rgb_tuple(*COLOR_HLS_VALUES[i]) for i in range(16)}}

class VideoFileConverterApp:
    def __init__(self, root):
        self.root, self.mode_var = root, tk.StringVar(value="pixel")
        self.use_hdr_var, self.use_error_correction_var, self.force_gpu_var, self.color_space_var = tk.BooleanVar(value=DEFAULTS['use_hdr']), tk.BooleanVar(value=DEFAULTS['use_error_correction']), tk.BooleanVar(value=CUDA_ENABLED), tk.StringVar(value=DEFAULTS['color_space'])
        self.setup_ui()
        self.root.title("File to Video Converter with Error Correction")

    def setup_ui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text="Convert File to Video with Reed-Solomon Error Correction:").pack(pady=10)
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(pady=5)
        ttk.Radiobutton(mode_frame, text="Pixel Mode", variable=self.mode_var, value="pixel", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Text Mode", variable=self.mode_var, value="text", command=self.update_ui).pack(side=tk.LEFT)
        self.config_frame = ttk.Frame(frame)
        self.config_frame.pack(fill='both', expand=True)
        self.update_ui()
        self.progress = ttk.Progressbar(frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=10)
        self.status_text = scrolledtext.ScrolledText(frame, height=10, width=70, state='disabled')
        self.status_text.pack(pady=10)
        self.log_message("Ready to encode...")
        ttk.Button(frame, text="Select File and Convert to Video", command=self.convert_file_to_video).pack(pady=5)

    def update_ui(self):
        for widget in self.config_frame.winfo_children(): widget.destroy()
        for name in DEFAULTS:
            if name not in ['use_hdr', 'use_error_correction', 'color_space', 'font_size', 'num_threads']:
                if self.mode_var.get() == "text" and name in ['pixel_block_width', 'pixel_block_height']: continue
                ttk.Label(self.config_frame, text=f"{name.replace('_', ' ').title()}:").pack()
                var = tk.IntVar(value=DEFAULTS[name])
                setattr(self, name, var)
                ttk.Entry(self.config_frame, textvariable=var).pack()
        ttk.Checkbutton(self.config_frame, text="Enable HDR", variable=self.use_hdr_var).pack(pady=5)
        ttk.Checkbutton(self.config_frame, text="Enable Error Correction", variable=self.use_error_correction_var).pack(pady=5)
        ttk.Checkbutton(self.config_frame, text="Force GPU", variable=self.force_gpu_var).pack(pady=5)
        ttk.Label(self.config_frame, text="Color Space:").pack()
        self.color_space_menu = ttk.Combobox(self.config_frame, textvariable=self.color_space_var, values=["bt709 (sRGB)", "bt2020 (HDR)"])
        self.color_space_menu.pack(pady=5)
        if self.mode_var.get() == "text":
            ttk.Label(self.config_frame, text="Font Size:").pack()
            self.font_size = tk.IntVar(value=DEFAULTS['font_size'])
            ttk.Entry(self.config_frame, textvariable=self.font_size).pack()
        ttk.Label(self.config_frame, text="Number of Threads:").pack()
        self.num_threads = tk.IntVar(value=DEFAULTS['num_threads'])
        ttk.Entry(self.config_frame, textvariable=self.num_threads).pack()

    def log_message(self, message):
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message + '\n')
        self.status_text.config(state='disabled')
        self.status_text.yview(tk.END)
        self.trim_log()

    def trim_log(self, max_lines=100):
        log_lines = self.status_text.get('1.0', tk.END).split('\n')
        if len(log_lines) > max_lines: self.status_text.delete('1.0', f'{len(log_lines) - max_lines}.0')

    def convert_file_to_video(self):
        file_path = filedialog.askopenfilename()
        output_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if not file_path or not output_path: return
        self.log_message(f"Selected file: {file_path}")
        self.log_message(f"Output path: {output_path}")
        if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path))
        threading.Thread(target=self.process_file_to_video, args=(file_path, output_path), daemon=True).start()

    def process_file_to_video(self, file_path, output_path):
        try:
            for attr in DEFAULTS:
                if attr not in ['use_hdr', 'use_error_correction', 'color_space', 'font_size', 'num_threads']: setattr(self, attr, getattr(self, attr).get())
            use_hdr, use_error_correction = self.use_hdr_var.get(), self.use_error_correction_var.get()
            color_space = self.color_space_var.get()
            font_size = self.font_size.get() if self.mode_var.get() == "text" else None
            num_threads = self.num_threads.get()
            force_gpu = self.force_gpu_var.get()

            data = self.read_file_data(file_path)
            if not data: return self.log_message("Error: No data read from file.")
            self.log_message(f"File read successfully. Size: {len(data)} bytes.")
            
            if use_error_correction:
                rs_n, rs_k = 255, 255 - (255 * self.error_correction_percentage.get() // 100)
                encoded_data = self.encode_data(data, rs_n, rs_k)
                if not encoded_data: return self.log_message("Error: Data encoding failed.")
                self.log_message(f"Data encoded successfully. Encoded size: {len(encoded_data)} bytes.")
            else:
                encoded_data = data

            if self.mode_var.get() == "pixel":
                self.create_video_pixel_mode(encoded_data, output_path, use_hdr, color_space, num_threads, force_gpu)
            else:
                self.create_video_text_mode(encoded_data, output_path, use_hdr, color_space, font_size, num_threads, force_gpu)
            self.log_message("Video created successfully.")
            self.progress.config(value=100)
        except Exception as e:
            self.log_message(f"Error: {str(e)}")
            self.progress.config(value=0)

    def read_file_data(self, file_path):
        self.log_message("Reading file data...")
        with open(file_path, 'rb') as file:
            data = file.read()
        self.log_message(f"File read: {len(data)} bytes")
        return data

    def encode_data(self, data, rs_n, rs_k):
        self.log_message(f"Encoding data with Reed-Solomon (rs_n={rs_n}, rs_k={rs_k})...")
        rs = RSCodec(rs_n - rs_k)
        try:
            encoded_data = rs.encode(data)
            self.log_message(f"Data encoded successfully: {len(encoded_data)} bytes")
            return encoded_data
        except ReedSolomonError as e:
            self.log_message(f"Reed-Solomon encoding error: {str(e)}")
            return None

    def create_video_pixel_mode(self, data, output_path, use_hdr, color_space, num_threads, force_gpu):
        self.log_message("Starting video creation in pixel mode...")
        frame_size = (self.video_width // self.pixel_block_width) * (self.video_height // self.pixel_block_height)
        num_frames = (len(data) + frame_size - 1) // frame_size
        padded_data, total_frames = data + bytes([0] * (num_frames * frame_size - len(data))), num_frames + 2
        self.log_message(f"Video will have {total_frames} frames (including start and end frames).")
        try:
            pix_fmt, vcodec = ('yuv420p10le', 'libx265') if use_hdr else ('yuv420p', 'libx264')
            color_primaries, color_trc, colorspace = ('bt2020', 'smpte2084', 'bt2020nc') if color_space == 'bt2020 (HDR)' else ('bt709', 'bt709', 'bt709')
            
            process = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.video_width}x{self.video_height}', framerate=self.fps)
                .output(output_path, pix_fmt=pix_fmt, vcodec=vcodec, preset='veryslow', crf=0, video_bitrate='2000k', maxrate='2000k', bufsize='4000k', color_primaries=color_primaries, color_trc=color_trc, colorspace=colorspace, rc_lookahead=32)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['START']).tobytes())
            self.log_message("Writing start frame...")

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_frame, padded_data[i * frame_size:(i + 1) * frame_size], force_gpu): i for i in range(num_frames)}
                results = [None] * num_frames
                for future in as_completed(futures):
                    i, frame_data = futures[future], future.result()
                    results[i] = frame_data
                    self.log_message(f"Processed frame {i + 1}/{total_frames}.")

            for i, frame_data in enumerate(results):
                process.stdin.write(frame_data)
                self.progress.config(value=(i + 1) / total_frames * 100)
                self.root.update_idletasks()

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("Writing end frame...")
            process.stdin.close()
            process.wait()
            self.log_message("Video processing completed.")
        except Exception as e:
            self.log_message(f"Error during video creation: {str(e)}")

    def create_video_text_mode(self, data, output_path, use_hdr, color_space, font_size, num_threads, force_gpu):
        self.log_message("Starting video creation in text mode...")
        hex_data = data.hex().upper()
        num_frames = (len(hex_data) + (self.video_width * self.video_height // (font_size * font_size)) - 1) // (self.video_width * self.video_height // (font_size * font_size))
        padded_data, total_frames = hex_data.ljust(num_frames * self.video_width * self.video_height // (font_size * font_size), '0'), num_frames + 2
        self.log_message(f"Video will have {total_frames} frames (including start and end frames).")
        try:
            pix_fmt, vcodec = ('yuv420p10le', 'libx265') if use_hdr else ('yuv420p', 'libx264')
            color_primaries, color_trc, colorspace = ('bt2020', 'smpte2084', 'bt2020nc') if color_space == 'bt2020 (HDR)' else ('bt709', 'bt709', 'bt709')
            
            process = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{self.video_width}x{self.video_height}', framerate=self.fps)
                .output(output_path, pix_fmt=pix_fmt, vcodec=vcodec, preset='veryslow', crf=0, video_bitrate='2000k', maxrate='2000k', bufsize='4000k', color_primaries=color_primaries, color_trc=color_trc, colorspace=colorspace, rc_lookahead=32)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['START']).tobytes())
            self.log_message("Writing start frame...")

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_text_frame, padded_data[i * (self.video_width * self.video_height // (font_size * font_size)):(i + 1) * (self.video_width * self.video_height // (font_size * font_size))], font_size, force_gpu): i for i in range(num_frames)}
                results = [None] * num_frames
                for future in as_completed(futures):
                    i, frame_data = futures[future], future.result()
                    results[i] = frame_data
                    self.log_message(f"Processed frame {i + 1}/{total_frames}.")

            for i, frame_data in enumerate(results):
                process.stdin.write(frame_data)
                self.progress.config(value=(i + 1) / total_frames * 100)
                self.root.update_idletasks()

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("Writing end frame...")
            process.stdin.close()
            process.wait()
            self.log_message("Video processing completed.")
        except Exception as e:
            self.log_message(f"Error during video creation: {str(e)}")

    def process_frame(self, frame_data, force_gpu):
        try:
            if CUDA_ENABLED and force_gpu:
                image = self.data_to_image_cuda(frame_data)
            else:
                image = self.data_to_image(frame_data)
            return image.tobytes()
        except Exception as e:
            self.log_message(f"Error processing frame: {str(e)}")
            raise

    def process_text_frame(self, frame_data, font_size, force_gpu):
        try:
            image = self.text_to_image(frame_data, font_size)
            return image.tobytes()
        except Exception as e:
            self.log_message(f"Error processing text frame: {str(e)}")
            raise

    def create_solid_color_frame(self, width, height, color):
        return np.full((height, width, 3), color, dtype=np.uint8)

    def data_to_image(self, data):
        image = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        blocks_per_row = self.video_width // self.pixel_block_width
        for i in range(len(data)):
            block_x = (i % blocks_per_row) * self.pixel_block_width
            block_y = (i // blocks_per_row) * self.pixel_block_height
            color = COLORS['MAP']['{:X}'.format(data[i] % 16)]
            image[block_y:block_y + self.pixel_block_height, block_x:block_x + self.pixel_block_width] = color
        return image

    def data_to_image_cuda(self, data):
        blocks_per_row = self.video_width // self.pixel_block_width
        image = cp.zeros((self.video_height, self.video_width, 3), dtype=cp.uint8)
        for i in range(len(data)):
            block_x = (i % blocks_per_row) * self.pixel_block_width
            block_y = (i // blocks_per_row) * self.pixel_block_height
            color = cp.array(COLORS['MAP']['{:X}'.format(data[i] % 16)], dtype=cp.uint8)
            image[block_y:block_y + self.pixel_block_height, block_x:block_x + self.pixel_block_width] = color
        return cp.asnumpy(image)

    def text_to_image(self, text, font_size):
        image = Image.new('RGB', (self.video_width, self.video_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        x, y = 0, 0
        for char in text:
            draw.text((x, y), char, fill=(0, 0, 0), font=font)
            x += font_size
            if x >= self.video_width:
                x = 0
                y += font_size
        return np.array(image)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFileConverterApp(root)
    root.mainloop()
