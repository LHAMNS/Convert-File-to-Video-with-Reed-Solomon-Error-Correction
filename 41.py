import os, shutil, tkinter as tk, threading, numpy as np, qrcode, ffmpeg
from tkinter import filedialog, ttk, scrolledtext
from reedsolo import RSCodec, ReedSolomonError
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorsys import hls_to_rgb
import queue
import cv2

CUDA_ENABLED = True


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
    'memory_cache_size': 4  # 默认缓存大小4GB
}

COLOR_HLS_VALUES = [(i / 16, 0.5, 1) for i in range(16)]
COLORS = {
    'START': (255, 255, 0),
    'END': (255, 0, 255),
    'MAP': {f'{i:X}': tuple(int(c * 255) for c in hls_to_rgb(*COLOR_HLS_VALUES[i])) for i in range(16)}
}

def clear_directory(dir_path):
    """清空指定目录，如果目录不存在则创建它"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

class VideoFileConverterApp:
    def __init__(self, root):
        self.root = root
        self.mode_var = tk.StringVar(value="pixel")
        self.use_hdr_var = tk.BooleanVar(value=DEFAULTS['use_hdr'])
        self.use_error_correction_var = tk.BooleanVar(value=DEFAULTS['use_error_correction'])
        self.force_gpu_var = tk.BooleanVar(value=CUDA_ENABLED)
        self.color_space_var = tk.StringVar(value=DEFAULTS['color_space'])
        self.auto_adjust_var = tk.BooleanVar(value=True)
        self.use_memory_cache_var = tk.BooleanVar(value=DEFAULTS['use_memory_cache'])
        self.memory_cache_size_var = tk.StringVar(value=str(DEFAULTS['memory_cache_size']))
        self.setup_ui()
        self.root.title("文件转视频工具，支持里德-所罗门纠错")
        clear_directory('cache')
        clear_directory('string')
        self.image_cache = queue.Queue()
        self.cache_size = 0
        self.stop_saving_thread = False
        threading.Thread(target=self.save_images_to_disk, daemon=True).start()

    def setup_ui(self):
        """设置图形界面"""
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill='both', expand=True)
        ttk.Label(frame, text="将文件转换为带有里德-所罗门纠错的视频:").pack(pady=10)
        
        # 模式选择按钮
        mode_frame = ttk.Frame(frame)
        mode_frame.pack(pady=5)
        ttk.Radiobutton(mode_frame, text="像素模式", variable=self.mode_var, value="pixel", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="文本模式", variable=self.mode_var, value="text", command=self.update_ui).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="二维码模式", variable=self.mode_var, value="qrcode", command=self.update_ui).pack(side=tk.LEFT)
        
        # 配置参数输入框
        self.config_frame = ttk.Frame(frame)
        self.config_frame.pack(fill='both', expand=True)
        self.update_ui()
        
        # 进度条
        self.progress = ttk.Progressbar(frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=10)
        
        # 状态日志
        self.status_text = scrolledtext.ScrolledText(frame, height=10, width=70, state='disabled')
        self.status_text.pack(pady=10)
        self.log_message("准备编码...")
        
        # 开始转换按钮
        ttk.Button(frame, text="选择文件并转换为视频", command=self.convert_file_to_video).pack(pady=5)

    def update_ui(self):
        """更新图形界面，显示或隐藏不同模式的输入框"""
        for widget in self.config_frame.winfo_children():
            widget.destroy()
        for name in DEFAULTS:
            if name not in ['use_hdr', 'use_error_correction', 'color_space', 'font_size', 'num_threads', 'use_memory_cache', 'memory_cache_size']:
                if self.mode_var.get() in ["text", "qrcode"] and name in ['pixel_block_width', 'pixel_block_height']:
                    continue
                ttk.Label(self.config_frame, text=f"{name.replace('_', ' ').title()}:").pack()
                var = tk.StringVar(value=str(DEFAULTS[name]))
                setattr(self, name, var)
                ttk.Entry(self.config_frame, textvariable=var).pack()
        
        # 特定模式的配置
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
        ttk.Label(self.config_frame, text="线程数量:").pack()
        self.num_threads = tk.StringVar(value=str(DEFAULTS['num_threads']))
        ttk.Entry(self.config_frame, textvariable=self.num_threads).pack()

    def toggle_qr_options(self):
        """根据自动调整二维码大小的选项，显示或隐藏手动调整的输入框"""
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
        """根据启用内存缓存的选项，显示或隐藏缓存大小输入框"""
        if self.use_memory_cache_var.get():
            self.memory_cache_size_label.pack(pady=5)
            self.memory_cache_size_entry.pack(pady=5)
        else:
            self.memory_cache_size_label.pack_forget()
            self.memory_cache_size_entry.pack_forget()

    def log_message(self, message):
        """记录日志信息"""
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message + '\n')
        self.status_text.config(state='disabled')
        self.status_text.yview(tk.END)
        self.trim_log()

    def trim_log(self, max_lines=100):
        """限制日志显示的最大行数"""
        log_lines = self.status_text.get('1.0', tk.END).split('\n')
        if len(log_lines) > max_lines:
            self.status_text.delete('1.0', f'{len(log_lines) - max_lines}.0')

    def convert_file_to_video(self):
        """选择文件并开始转换为视频"""
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
        """处理文件并转换为视频"""
        try:
            # 获取并转换用户输入的参数
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
            self.memory_cache_size = int(self.memory_cache_size_var.get()) * 1024 * 1024 * 1024  # 转换为字节

            # 读取文件数据
            data = self.read_file_data(file_path)
            if not data:
                return self.log_message("错误: 无法读取文件数据。")
            self.log_message(f"文件读取成功，大小: {len(data)} 字节。")
            
            # 数据编码
            if self.use_error_correction_var.get():
                rs_n, rs_k = 255, 255 - (255 * self.error_correction_percentage // 100)
                encoded_data = self.encode_data(data, rs_n, rs_k)
                if not encoded_data:
                    return self.log_message("错误: 数据编码失败。")
                self.log_message(f"数据编码成功，编码后大小: {len(encoded_data)} 字节。")
            else:
                encoded_data = data

            # 根据选择的模式创建视频
            if self.mode_var.get() == "pixel":
                self.create_video_pixel_mode(encoded_data, output_path, self.use_hdr_var.get(), self.color_space_var.get(), self.num_threads, self.force_gpu_var.get())
            elif self.mode_var.get() == "text":
                self.create_video_text_mode(encoded_data, output_path, self.use_hdr_var.get(), self.color_space_var.get(), self.font_size, self.num_threads, self.force_gpu_var.get())
            elif self.mode_var.get() == "qrcode":
                self.create_video_qrcode_mode(encoded_data, output_path, self.use_hdr_var.get(), self.color_space_var.get(), self.num_threads, self.force_gpu_var.get())
            self.log_message("视频创建成功。")
            self.progress.config(value=100)
        except Exception as e:
            self.log_message(f"错误: {str(e)}")
            self.progress.config(value=0)
        finally:
            clear_directory('cache')
            clear_directory('string')
            self.stop_saving_thread = True

    def read_file_data(self, file_path):
        """读取文件数据"""
        self.log_message("读取文件数据...")
        with open(file_path, 'rb') as file:
            data = file.read()
        self.log_message(f"文件读取成功，大小: {len(data)} 字节")
        return data

    def encode_data(self, data, rs_n, rs_k):
        """使用里德-所罗门编码对数据进行编码"""
        self.log_message(f"使用里德-所罗门编码 (rs_n={rs_n}, rs_k={rs_k}) 对数据进行编码...")
        rs = RSCodec(rs_n - rs_k)
        try:
            encoded_data = rs.encode(data)
            self.log_message(f"数据编码成功: {len(encoded_data)} 字节")
            return encoded_data
        except ReedSolomonError as e:
            self.log_message(f"里德-所罗门编码错误: {str(e)}")
            return None

    def create_video_pixel_mode(self, data, output_path, use_hdr, color_space, num_threads, force_gpu):
        """创建像素模式的视频"""
        self.log_message("开始创建像素模式的视频...")
        frame_size = (self.video_width // self.pixel_block_width) * (self.video_height // self.pixel_block_height)
        num_frames = (len(data) + frame_size - 1) // frame_size
        padded_data = data + bytes([0] * (num_frames * frame_size - len(data)))
        total_frames = num_frames + 2  # 包含开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

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
            self.log_message("写入开始帧...")

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_frame, padded_data[i * frame_size:(i + 1) * frame_size], force_gpu): i for i in range(num_frames)}
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
                self.progress.config(value=(i + 1) / total_frames * 100)
                self.root.update_idletasks()

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def create_video_text_mode(self, data, output_path, use_hdr, color_space, font_size, num_threads, force_gpu):
        """创建文本模式的视频"""
        self.log_message("开始创建文本模式的视频...")
        hex_data = data.hex().upper()
        num_frames = (len(hex_data) + (self.video_width * self.video_height // (font_size * font_size)) - 1) // (self.video_width * self.video_height // (font_size * font_size))
        padded_data = hex_data.ljust(num_frames * self.video_width * self.video_height // (font_size * font_size), '0')
        total_frames = num_frames + 2  # 包括开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

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
            self.log_message("写入开始帧...")

            self.generate_text_images(font_size)

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_text_frame, padded_data[i * (self.video_width * self.video_height // (font_size * font_size)):(i + 1) * (self.video_width * self.video_height // (font_size * font_size))], font_size, force_gpu): i for i in range(num_frames)}
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
                self.progress.config(value=(i + 1) / total_frames * 100)
                self.root.update_idletasks()

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def create_video_qrcode_mode(self, data, output_path, use_hdr, color_space, num_threads, force_gpu):
        """创建二维码模式的视频"""
        self.log_message("开始创建二维码模式的视频...")
        chunk_size = 1024
        num_frames = (len(data) + chunk_size - 1) // chunk_size
        padded_data = data + bytes([0] * (num_frames * chunk_size - len(data)))
        total_frames = num_frames + 2  # 包括开始和结束帧
        self.log_message(f"视频将包含 {total_frames} 帧（包括开始和结束帧）。")

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
            self.log_message("写入开始帧...")

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(self.process_qrcode_frame, padded_data[i * chunk_size:(i + 1) * chunk_size], force_gpu): i for i in range(num_frames)}
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
                self.progress.config(value=(i + 1) / total_frames * 100)
                self.root.update_idletasks()

            process.stdin.write(self.create_solid_color_frame(self.video_width, self.video_height, COLORS['END']).tobytes())
            self.log_message("写入结束帧...")
            process.stdin.close()
            process.wait()
            self.log_message("视频处理完成。")
        except Exception as e:
            self.log_message(f"视频创建过程中出错: {str(e)}")

    def generate_text_images(self, font_size):
        """生成文本图像"""
        clear_directory('string')
        font = ImageFont.truetype("arial.ttf", font_size)
        for char in "0123456789ABCDEF":
            image = Image.new('RGB', (font_size, font_size), color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            w, h = draw.textbbox((0, 0), char, font=font)[2:]
            draw.text(((font_size - w) / 2, (font_size - h) / 2), char, font=font, fill=(0, 0, 0))
            image.save(f'string/{char}.png')

    def process_frame(self, frame_data, force_gpu):
        """处理像素模式的帧数据"""
        try:
            if CUDA_ENABLED and force_gpu:
                image = self.data_to_image_cuda(frame_data)
            else:
                image = self.data_to_image(frame_data)
            image_array = np.array(image)
            self.image_cache.put(image_array.tobytes())
            self.cache_size += image_array.nbytes
            if self.cache_size > self.memory_cache_size:
                self.log_message("缓存大小已达上限。等待磁盘写入...")
                while self.cache_size > self.memory_cache_size // 2:
                    pass  # 等待缓存减少到一半
            return image_array.tobytes()
        except Exception as e:
            self.log_message(f"处理帧时出错: {str(e)}")
            raise

    def process_text_frame(self, frame_data, font_size, force_gpu):
        """处理文本模式的帧数据"""
        try:
            image = Image.new('RGB', (self.video_width, self.video_height), color=(255, 255, 255))
            x, y = 0, 0
            for char in frame_data:
                char_image = Image.open(f'string/{char}.png')
                image.paste(char_image, (x, y))
                x += font_size
                if x >= self.video_width:
                    x = 0
                    y += font_size
            image_array = np.array(image)
            self.image_cache.put(image_array.tobytes())
            self.cache_size += image_array.nbytes
            if self.cache_size > self.memory_cache_size:
                self.log_message("缓存大小已达上限。等待磁盘写入...")
                while self.cache_size > self.memory_cache_size // 2:
                    pass  # 等待缓存减少到一半
            return image_array.tobytes()
        except Exception as e:
            self.log_message(f"处理文本帧时出错: {str(e)}")
            raise

    def process_qrcode_frame(self, frame_data, force_gpu):
        """处理二维码模式的帧数据"""
        try:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=1)
            qr.add_data(frame_data)
            qr.make(fit=True)
            qr_image = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            qr_image = qr_image.resize((self.video_width, self.video_height), Image.Resampling.LANCZOS)
            image_array = np.array(qr_image)
            self.image_cache.put(image_array.tobytes())
            self.cache_size += image_array.nbytes
            if self.cache_size > self.memory_cache_size:
                self.log_message("缓存大小已达上限。等待磁盘写入...")
                while self.cache_size > self.memory_cache_size // 2:
                    pass  # 等待缓存减少到一半
            return image_array.tobytes()
        except Exception as e:
            self.log_message(f"处理二维码帧时出错: {str(e)}")
            raise

    def save_images_to_disk(self):
        """将缓存中的图像保存到磁盘"""
        while not self.stop_saving_thread:
            if not self.image_cache.empty():
                image_data = self.image_cache.get()
                frame_path = os.path.join('cache', f'frame_{len(os.listdir("cache"))}.png')
                with open(frame_path, 'wb') as f:
                    f.write(image_data)
                self.cache_size -= len(image_data)
            else:
                threading.Event().wait(0.1)  # 等待一段时间再检查

    def create_solid_color_frame(self, width, height, color):
        """创建纯色帧"""
        return np.full((height, width, 3), color, dtype=np.uint8)

    def data_to_image(self, data):
        """将数据转换为图像"""
        image = np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)
        blocks_per_row = self.video_width // self.pixel_block_width
        for i in range(len(data)):
            block_x = (i % blocks_per_row) * self.pixel_block_width
            block_y = (i // blocks_per_row) * self.pixel_block_height
            color = COLORS['MAP']['{:X}'.format(data[i] % 16)]
            image[block_y:block_y + self.pixel_block_height, block_x:block_x + self.pixel_block_width] = color
        return image

    def data_to_image_cuda(self, data):
        """使用CUDA将数据转换为图像"""
        blocks_per_row = self.video_width // self.pixel_block_width
        image = cp.zeros((self.video_height, self.video_width, 3), dtype=cp.uint8)
        for i in range(len(data)):
            block_x = (i % blocks_per_row) * self.pixel_block_width
            block_y = (i // blocks_per_row) * self.pixel_block_height
            color = cp.array(COLORS['MAP']['{:X}'.format(data[i] % 16)])
            image[block_y:block_y + self.pixel_block_height, block_x:block_x + self.pixel_block_width] = color
        return image.get()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoFileConverterApp(root)
    root.mainloop()
