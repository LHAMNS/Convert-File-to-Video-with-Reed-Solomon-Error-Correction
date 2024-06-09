import tkinter as tk
from tkinter import scrolledtext
import threading

class LogViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("日志查看器")
        self.setup_ui()
        self.stop_thread = False
        self.log_file = "log.txt"
        threading.Thread(target=self.update_log, daemon=True).start()

    def setup_ui(self):
        """Setup the UI."""
        frame = tk.Frame(self.root, padding="10")
        frame.pack(fill='both', expand=True)

        # Progress bar
        self.progress = tk.ttk.Progressbar(frame, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=10)

        # Status log
        self.status_text = scrolledtext.ScrolledText(frame, height=20, width=80, state='disabled')
        self.status_text.pack(pady=10)

        # Frequency preview grid
        self.preview_frame = tk.Frame(frame)
        self.preview_frame.pack(pady=10)
        self.preview_labels = []

    def update_log(self):
        """Update the log display."""
        while not self.stop_thread:
            try:
                with open(self.log_file, 'r') as log_file:
                    lines = log_file.readlines()
                self.status_text.config(state='normal')
                self.status_text.delete('1.0', tk.END)
                self.status_text.insert(tk.END, ''.join(lines))
                self.status_text.config(state='disabled')
                self.status_text.yview(tk.END)
                self.update_preview(lines[-1])
            except Exception as e:
                pass
            finally:
                threading.Event().wait(1)

    def update_preview(self, line):
        """Update the frequency preview grid."""
        if "处理音频帧" in line:
            frame_info = line.split(' ')
            frame_num = int(frame_info[2].split('/')[0])
            total_frames = int(frame_info[2].split('/')[1])
            self.progress.config(value=(frame_num / total_frames) * 100)

if __name__ == "__main__":
    root = tk.Tk()
    app = LogViewerApp(root)
    root.mainloop()
