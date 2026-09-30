import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import serial
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import pywt
import time
import threading
import pandas as pd
from serial.tools import list_ports

# --- Konfigurasi Port Serial (Initial / Default values) ---
SERIAL_PORT = 'COM1'    # Default COM port, akan diupdate dari dropdown
BAUD_RATE = 9600    # PASTIKAN INI SAMA DENGAN BAUD RATE ARDUINO ANDA
SAMPLE_RATE = 100   # Hz (sampling rate dari sensor ECG/Arduino Anda), sesuaikan dengan fs di kode drop excel
NYQUIST_RATE = SAMPLE_RATE / 2 
BUFFER_SIZE = 300   # Jumlah titik data yang ditampilkan pada plot (sekitar 3 detik data pada 100 Hz)
PLOT_INTERVAL = 50   # Interval update plot dalam milidetik (lebih kecil = lebih cepat, lebih besar = lebih lambat)

# --- Variabel Global (Accessed by multiple threads, so protect with lock) ---
ecg_buffer = np.zeros(BUFFER_SIZE)     
time_vector = np.arange(BUFFER_SIZE) / SAMPLE_RATE 

is_running = False  
is_paused = False   

signals_buffer = {}     # Dictionary untuk menyimpan berbagai versi sinyal yang diproses
p_peaks_buffer, q_peaks_buffer, r_peaks_buffer, s_peaks_buffer, t_peaks_buffer = [], [], [], [], []

noise_reduction_percentage = "N/A"
bpm_value = "N/A"

serial_connection = None 
data_lock = threading.Lock() 

all_raw_data = [] 

class RealTimeECGViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time ECG Viewer")
        self.root.geometry('1300x950')
        self.root.configure(bg='#e6f2ff') 

        self.serial_thread = None 
        self.is_connected = False 
        self.selected_filter = tk.StringVar(value='Detected PQRST') # Default filter sesuai kode drop excel

        # Selalu tampilkan COM1 hingga COM10 sebagai pilihan awal
        available_ports = [f'COM{i}' for i in range(1, 12)] 
        
        # Tambahkan port yang terdeteksi secara otomatis jika belum ada dalam daftar
        detected_ports = [port.device for port in list_ports.comports()]
        for port in detected_ports:
            if port not in available_ports:
                available_ports.append(port)
        
        self.selected_com_port = tk.StringVar(value=available_ports[0] if available_ports else 'COM1') 

        top_controls_frame = tk.Frame(root, bg='#e6f2ff')
        top_controls_frame.pack(pady=10)

        titles = ['Original ECG', 'Bandpass Filtered', 'Baseline Corrected',
                    'Wavelet Denoised', 'Smoothed Signal', 'Detected PQRST', 'All Signals']
        
        dropdown_filter = ttk.Combobox(top_controls_frame, textvariable=self.selected_filter, values=titles,
                                        font=('Arial', 12), state='readonly', width=18)
        dropdown_filter.pack(side=tk.LEFT, padx=10)
        dropdown_filter.bind('<<ComboboxSelected>>', self.change_filter)

        self.com_port_dropdown = ttk.Combobox(top_controls_frame, textvariable=self.selected_com_port, values=available_ports,
                                                font=('Arial', 12), state='readonly', width=10)
        self.com_port_dropdown.pack(side=tk.LEFT, padx=10)
        self.com_port_dropdown.bind('<<ComboboxSelected>>', self.update_serial_port_setting) 
        
        self.bpm_label = tk.Label(root, text="BPM: N/A", font=('Arial', 14), bg='#e6f2ff')
        self.bpm_label.pack(pady=5)
        self.noise_label = tk.Label(root, text="Noise Reduction: N/A", font=('Arial', 14), bg='#e6f2ff')
        self.noise_label.pack(pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 6)) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=root) 
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True) 

        self.lines = {}
        for key in titles:
            if key != 'All Signals':
                self.lines[key], = self.ax.plot([], [], label=key, color='black', linewidth=1)
        
        self.peak_markers = {
            'P': None, 'Q': None, 'R': None, 'S': None, 'T': None
        }

        self.result_text = tk.Text(root, height=12, font=('Courier New', 10), bg='white')
        self.result_text.pack(fill=tk.X, padx=10, pady=5)

        ctrl = tk.Frame(root, bg='#e6f2ff')
        ctrl.pack(pady=10)

        self.connect_btn = tk.Button(ctrl, text='🔌 Connect', command=self.connect_serial,
                                        font=('Arial', 12), bg='#4CAF50', fg='white')
        self.connect_btn.pack(side=tk.LEFT, padx=5)

        self.disconnect_btn = tk.Button(ctrl, text='❌ Disconnect', command=self.disconnect_serial,
                                            font=('Arial', 12), bg='#f44336', fg='white', state=tk.DISABLED)
        self.disconnect_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(ctrl, text='⏸️ Pause', command=self.toggle_pause,
                                    font=('Arial', 12), bg='#2196F3', fg='white')
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        reset_btn = tk.Button(ctrl, text='🔄 Reset Pengujian', command=self.reset_testing,
                                font=('Arial', 12), bg='#ff9800', fg='white')
        reset_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(ctrl, text='💾 Save Image', command=self.save_image,
                                            font=('Arial', 12), bg='#673ab7', fg='white')
        save_btn.pack(side=tk.LEFT, padx=5)

        save_result_btn = tk.Button(ctrl, text='📸 Save Result', command=self.save_result_image,
                                                font=('Arial', 12), bg='#3f51b5', fg='white')
        save_result_btn.pack(side=tk.LEFT, padx=5)

        save_comparison_btn = tk.Button(ctrl, text='📊 Save Comparison', command=self.save_comparison_excel_realtime,
                                            font=('Arial', 12), bg='#008080', fg='white', width=18)
        save_comparison_btn.pack(side=tk.LEFT, padx=5)

        speed_slider = tk.Scale(ctrl, from_=0.5, to=3.0, resolution=0.1,
                                            orient=tk.HORIZONTAL, label='Speed',
                                            command=self.change_speed, bg='#e6f2ff')
        speed_slider.set(1.0)
        speed_slider.pack(side=tk.LEFT, padx=5)

        self.ani = FuncAnimation(self.fig, self.update_plot, interval=PLOT_INTERVAL, cache_frame_data=False)

    def update_serial_port_setting(self, event=None):
        global SERIAL_PORT 
        SERIAL_PORT = self.selected_com_port.get()
        print(f"Port COM diubah menjadi: {SERIAL_PORT}")

    def ensure_positive_and_round(self, signal_data, original_data):
        """
        Memastikan sinyal positif dan membulatkan ke 2 desimal.
        Fungsi ini disalin dari kode drop Excel.
        """
        min_original = np.min(original_data) if original_data.size > 0 else 0
        epsilon = 1e-9
        processed_signal_data = signal_data.copy()
        if min_original < 0:
            processed_signal_data += abs(min_original) + epsilon
        elif np.any(processed_signal_data <= 0):
            processed_signal_data += epsilon
        return np.round(processed_signal_data, 2)

    def calculate_bpm(self, r_peak_indices, sampling_rate):
        if len(r_peak_indices) < 2:
            return "N/A"
        
        intervals = np.diff(r_peak_indices) / sampling_rate 
        
        if np.any(intervals <= 0):
            return "Error: Non-positive interval"
        
        mean_interval = np.mean(intervals)
        bpm = 60 / mean_interval
        return f"{bpm:.2f}"

    def calculate_noise_reduction(self, original_segment, processed_segment):
        """
        Menghitung persentase pengurangan noise menggunakan sum square (sesuai kode drop Excel).
        """
        min_len = min(len(original_segment), len(processed_segment))
        original_segment = original_segment[:min_len]
        processed_segment = processed_segment[:min_len] # Pastikan ini adalah sinyal setelah smoothing dan ensure_positive_and_round

        if not original_segment.size or not processed_segment.size:
            return "N/A"
        
        original_power = np.sum(original_segment**2)
        smoothed_power = np.sum(processed_segment**2) 

        if original_power > 0:
            reduction_percentage = 100 * (1 - (smoothed_power / original_power))
            return f"{reduction_percentage:.2f}%"
        else:
            return "0.00%" 

    def process_ecg_realtime(self, ecg_data_raw):
        """
        Melakukan pemrosesan sinyal ECG: filtering, baseline correction, denoising, smoothing,
        dan deteksi puncak PQRST. Disesuaikan agar sama dengan kode drop Excel.
        """
        if not ecg_data_raw.size:
            return {}
        ecg_data_float = ecg_data_raw.astype(float)

        processed_results = {'Original ECG': ecg_data_float}

        # 1. Bandpass Filtering (Butterworth 6th order, 0.5 - 40 Hz)
        b_bp, a_bp = signal.butter(6, [0.5 / NYQUIST_RATE, 40 / NYQUIST_RATE], btype='band')
        filtered = signal.filtfilt(b_bp, a_bp, ecg_data_float)
        filtered_positive = self.ensure_positive_and_round(filtered, ecg_data_float)
        processed_results['Bandpass Filtered'] = filtered_positive

        # 2. Baseline Wander Correction (Gaussian filter)
        baseline = gaussian_filter1d(filtered, sigma=int(SAMPLE_RATE * 0.3 / 6)) # fs * 0.3 / 6
        corrected = filtered - baseline
        corrected_positive = self.ensure_positive_and_round(corrected, ecg_data_float)
        processed_results['Baseline Corrected'] = corrected_positive

        # 3. Wavelet Denoising (menggunakan 'coif4' wavelet pada level 5)
        wavelet = 'coif4' 
        level = 5 
        try:
            coeffs = pywt.wavedec(corrected, wavelet, level=level)
            thresholded_coeffs = [coeffs[0]] + [pywt.threshold(c, np.median(np.abs(c)) / 0.6745, 'soft') for c in coeffs[1:]]
            denoised = pywt.waverec(thresholded_coeffs, wavelet)[:len(corrected)]
        except ValueError as e:
            print(f"Wavelet denoising error: {e}. Returning corrected signal instead.")
            denoised = corrected
        except Exception as e:
            print(f"An unexpected error occurred during wavelet denoising: {e}. Returning corrected signal.")
            denoised = corrected

        denoised_modified = denoised - 6 # Modifikasi ini penting dari kode drop Excel
        denoised_positive = self.ensure_positive_and_round(denoised_modified, ecg_data_float)
        processed_results['Wavelet Denoised'] = denoised_positive

        # 4. Smoothing (Bartlett window, 0.05 * fs)
        window_size_smooth = int(SAMPLE_RATE * 0.05)
        smoothed = signal.convolve(denoised_modified, signal.windows.bartlett(window_size_smooth), mode='same')
        smoothed_positive = self.ensure_positive_and_round(smoothed, ecg_data_float)
        processed_results['Smoothed Signal'] = smoothed_positive
        
        # 5. Normalisasi (Z-score)
        if np.std(smoothed) > 1e-9: # Hindari pembagian dengan nol
            normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)
        else:
            normalized = np.zeros_like(smoothed)
        
        normalized_positive = self.ensure_positive_and_round(normalized, ecg_data_float)
        processed_results['Detected PQRST'] = normalized_positive

        # --- Deteksi PQRST (disesuaikan dengan jendela dan kriteria kode drop Excel) ---
        
        # Deteksi R-peak
        r_peaks_temp, _ = signal.find_peaks(normalized, height=0.5, distance=int(0.5 * SAMPLE_RATE))
        
        p_peaks_temp, q_peaks_temp, s_peaks_temp, t_peaks_temp = [], [], [], []

        for r_idx in r_peaks_temp:
            # 6. Deteksi Q-peak
            q_search_start = max(0, r_idx - int(0.1 * SAMPLE_RATE)) 
            q_search_end = r_idx 
            q_peak_idx = -1
            if q_search_start < q_search_end:
                segment_q = normalized[q_search_start:q_search_end]
                if segment_q.size > 0:
                    q_peak_idx = np.argmin(segment_q) + q_search_start
            q_peaks_temp.append(q_peak_idx)

            # 7. Deteksi S-peak
            s_search_start = r_idx 
            s_search_end = min(len(normalized), r_idx + int(0.1 * SAMPLE_RATE)) 
            s_peak_idx = -1
            if s_search_start < s_search_end:
                segment_s = normalized[s_search_start:s_search_end]
                if segment_s.size > 0:
                    s_peak_idx = np.argmin(segment_s) + s_search_start
            s_peaks_temp.append(s_peak_idx)

            # 8. Deteksi P-peak
            p_search_start = max(0, q_search_start - int(0.2 * SAMPLE_RATE))
            p_search_end = q_search_start
            p_peak_idx = -1
            if p_search_start < p_search_end:
                segment_p = normalized[p_search_start:p_search_end]
                if segment_p.size > 0:
                    p_peak_idx = np.argmax(segment_p) + p_search_start
            p_peaks_temp.append(p_peak_idx)
            
            # 9. Deteksi T-peak
            t_search_start = s_search_end
            t_search_end = min(len(normalized), s_search_end + int(0.4 * SAMPLE_RATE)) 
            t_peak_idx = -1
            if t_search_start < t_search_end:
                segment_t = normalized[t_search_start:t_search_end]
                if segment_t.size > 0:
                    t_peak_idx = np.argmax(segment_t) + t_search_start
            t_peaks_temp.append(t_peak_idx)

        processed_results['r_peaks'] = np.array([idx for idx in r_peaks_temp if idx != -1])
        processed_results['q_peaks'] = np.array([idx for idx in q_peaks_temp if idx != -1])
        processed_results['s_peaks'] = np.array([idx for idx in s_peaks_temp if idx != -1])
        processed_results['p_peaks'] = np.array([idx for idx in p_peaks_temp if idx != -1])
        processed_results['t_peaks'] = np.array([idx for idx in t_peaks_temp if idx != -1])

        return processed_results

    def read_serial(self):
        global ecg_buffer, is_running, serial_connection, all_raw_data

        try:
            serial_connection = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) 
            print(f"Terhubung ke {SERIAL_PORT} dengan baud rate {BAUD_RATE}")
            
            self.is_connected = True 
            self.root.after(0, self.update_connection_status_gui) 

            while is_running and self.is_connected and serial_connection and serial_connection.is_open:
                try:
                    line = serial_connection.readline().decode('utf-8').strip()
                    if line: 
                        val = int(line)
                        with data_lock: 
                            ecg_buffer = np.roll(ecg_buffer, -1) 
                            ecg_buffer[-1] = val 
                            all_raw_data.append(val) 
                except ValueError:
                    pass 
                except UnicodeDecodeError as e:
                    print(f"Error decoding serial data: {e}")
                except serial.SerialTimeoutException:
                    pass 
                except serial.SerialException as e:
                    print(f"Serial communication error in read_serial loop: {e}")
                    is_running = False 
                    self.root.after(0, lambda: messagebox.showerror("Serial Error", f"Komunikasi serial terputus:\n{e}"))
                    self.root.after(0, self.disconnect_serial) 
                time.sleep(0.001) 

        except serial.SerialException as e:
            print(f"Gagal terhubung ke {SERIAL_PORT} pada tahap awal: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Gagal terhubung ke {SERIAL_PORT}:\n{e}\nPastikan port tersedia dan tidak digunakan oleh aplikasi lain.")) #
            self.is_connected = False 
            self.root.after(0, self.update_connection_status_gui)
        finally:
            if serial_connection and serial_connection.is_open:
                serial_connection.close()
                print("Koneksi serial ditutup dari finally block.")
            is_running = False 
            self.is_connected = False 
            self.root.after(0, self.update_connection_status_gui) 

    def update_connection_status_gui(self):
        self.connect_btn.config(state=tk.NORMAL if not self.is_connected else tk.DISABLED)
        self.disconnect_btn.config(state=tk.DISABLED if not self.is_connected else tk.NORMAL)
        self.com_port_dropdown.config(state='readonly' if not self.is_connected else tk.DISABLED)

    def update_plot(self, frame):
        global ecg_buffer, signals_buffer, p_peaks_buffer, q_peaks_buffer, r_peaks_buffer, s_peaks_buffer, t_peaks_buffer, bpm_value, noise_reduction_percentage

        if not is_paused and self.is_connected:
            with data_lock:
                current_ecg_data = ecg_buffer.copy()

            if current_ecg_data.any():
                processed = self.process_ecg_realtime(current_ecg_data)
                
                signals_buffer.update({k: v for k, v in processed.items() if k not in ['r_peaks', 'q_peaks', 's_peaks', 'p_peaks', 't_peaks']})
                
                self.update_pqrst_analysis(processed) 

                if len(r_peaks_buffer) >= 2:
                    bpm_value = self.calculate_bpm(r_peaks_buffer, SAMPLE_RATE)
                    self.bpm_label.config(text=f"BPM: {bpm_value}")
                else:
                    self.bpm_label.config(text="BPM: N/A")

                if 'Original ECG' in processed and 'Smoothed Signal' in processed:
                    noise_reduction_percentage = self.calculate_noise_reduction(
                        processed['Original ECG'],
                        processed['Smoothed Signal'] 
                    )
                    self.noise_label.config(text=f"Noise Reduction: {noise_reduction_percentage}")
                else:
                    self.noise_label.config(text="Noise Reduction: N/A")
            else:
                self.bpm_label.config(text="BPM: N/A")
                self.noise_label.config(text="Noise Reduction: N/A")
        
        self.fig.clear() 
        choice = self.selected_filter.get()

        if choice == 'All Signals':
            keys_to_plot = ['Original ECG', 'Bandpass Filtered', 'Baseline Corrected',
                            'Wavelet Denoised', 'Smoothed Signal', 'Detected PQRST']
            gs = gridspec.GridSpec(3, 2, figure=self.fig, hspace=0.6, wspace=0.3)

            for i, key in enumerate(keys_to_plot):
                ax_sub = self.fig.add_subplot(gs[i])
                signal_to_plot = signals_buffer.get(key, np.array([]))

                ax_sub.plot(time_vector[:len(signal_to_plot)], signal_to_plot, label=key, linewidth=1, color='black')
                ax_sub.set_title(key, fontsize=10)
                ax_sub.grid(True)
                ax_sub.set_xlabel("Time (s)", fontsize=8)
                ax_sub.set_ylabel("Amplitude", fontsize=8)
                
                if signal_to_plot.size > 0:
                    y_min, y_max = np.min(signal_to_plot), np.max(signal_to_plot)
                    padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                    ax_sub.set_ylim(y_min - padding, y_max + padding)
                else:
                    ax_sub.set_ylim(-1, 1) 
                ax_sub.set_xlim(time_vector[0], time_vector[-1])
                ax_sub.tick_params(axis='both', which='major', labelsize=8)

                if key == 'Detected PQRST':
                    self.plot_peak(ax_sub, time_vector, signal_to_plot, p_peaks_buffer, 'm', 'P')
                    self.plot_peak(ax_sub, time_vector, signal_to_plot, q_peaks_buffer, 'c', 'Q') 
                    self.plot_peak(ax_sub, time_vector, signal_to_plot, r_peaks_buffer, 'r', 'R')
                    self.plot_peak(ax_sub, time_vector, signal_to_plot, s_peaks_buffer, 'b', 'S')
                    self.plot_peak(ax_sub, time_vector, signal_to_plot, t_peaks_buffer, 'g', 'T') 
                    ax_sub.legend(fontsize='x-small', loc='upper right')
        else:
            if not hasattr(self, 'ax') or not self.fig.axes:
                self.ax = self.fig.add_subplot(111) 
            else:
                self.ax.clear() 

            signal_to_plot = signals_buffer.get(choice, np.array([]))
            self.ax.plot(time_vector[:len(signal_to_plot)], signal_to_plot, color='black', linewidth=1)
            self.ax.set_title(choice)
            self.ax.grid(True)
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            
            if signal_to_plot.size > 0:
                y_min, y_max = np.min(signal_to_plot), np.max(signal_to_plot)
                padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                self.ax.set_ylim(y_min - padding, y_max + padding)
            else:
                self.ax.set_ylim(-1, 1) 
            self.ax.set_xlim(time_vector[0], time_vector[-1])

            if choice == 'Detected PQRST':
                self.plot_peak(self.ax, time_vector, signal_to_plot, p_peaks_buffer, 'm', 'P')
                self.plot_peak(self.ax, time_vector, signal_to_plot, q_peaks_buffer, 'c', 'Q')
                self.plot_peak(self.ax, time_vector, signal_to_plot, r_peaks_buffer, 'r', 'R')
                self.plot_peak(self.ax, time_vector, signal_to_plot, s_peaks_buffer, 'b', 'S')
                self.plot_peak(self.ax, time_vector, signal_to_plot, t_peaks_buffer, 'g', 'T')
                self.ax.legend(fontsize='small')

        self.canvas.draw_idle()

    @staticmethod
    def plot_peak(ax, time_vec, signal, peaks, color, label):
        valid_peaks = [p for p in peaks if 0 <= p < len(signal) and p != -1]
        if valid_peaks:
            ax.plot(time_vec[valid_peaks], signal[valid_peaks], marker='o', linestyle='', color=color, markersize=4, label=label)

    def update_pqrst_analysis(self, processed_signals):
        """
        Menganalisis deteksi PQRST dan memperbarui text output dengan detail per denyutan
        dan persentase akurasi, disesuaikan dengan logika kode drop Excel.
        """
        global p_peaks_buffer, q_peaks_buffer, r_peaks_buffer, s_peaks_buffer, t_peaks_buffer
        
        normalized_signal = processed_signals.get('Detected PQRST', np.array([]))
        r_peaks_temp_from_process = processed_signals.get('r_peaks', np.array([])) 

        self.result_text.delete('1.0', tk.END) #

        if not r_peaks_temp_from_process.size or not normalized_signal.size:
            self.result_text.insert(tk.END, "Tidak ada deteksi R-peak yang valid dalam jendela saat ini.\n")
            p_peaks_buffer[:], q_peaks_buffer[:], r_peaks_buffer[:], s_peaks_buffer[:], t_peaks_buffer[:] = [], [], [], [], []
            return

        total_beats = len(r_peaks_temp_from_process)
        p_temp, q_temp, r_temp, s_temp, t_temp = [], [], [], [], []
        valid_counts = {"P": 0, "Q": 0, "R": total_beats, "S": 0, "T": 0}

        for i, r_idx in enumerate(r_peaks_temp_from_process):
            r_time_ms = int(time_vector[r_idx] * 1000)

            line = f"Beat {i + 1}:"
            line += f" R: {r_time_ms} ms ✔"

            # Q-peak (sesuai jendela dan kriteria kode drop Excel)
            q_search_start = max(0, r_idx - int(0.1 * SAMPLE_RATE))
            q_search_end = r_idx
            q_idx_beat = -1
            if q_search_start < q_search_end and normalized_signal[q_search_start:q_search_end].size > 0:
                q_peak_idx_cand = np.argmin(normalized_signal[q_search_start:q_search_end]) + q_search_start
                if q_peak_idx_cand != -1 and (q_peak_idx_cand < r_idx) and (r_idx - q_peak_idx_cand <= int(0.08 * SAMPLE_RATE)):
                    q_idx_beat = q_peak_idx_cand
                    line += f" Q: {int(time_vector[q_idx_beat] * 1000)} ms ✔"
                    valid_counts["Q"] += 1
                else:
                    line += f" Q: ✘"
            else:
                line += f" Q: ✘"
            q_temp.append(q_idx_beat)

            # S-peak (sesuai jendela dan kriteria kode drop Excel)
            s_search_start = r_idx
            s_search_end = min(len(normalized_signal), r_idx + int(0.1 * SAMPLE_RATE))
            s_idx_beat = -1
            if s_search_start < s_search_end and normalized_signal[s_search_start:s_search_end].size > 0:
                s_peak_idx_cand = np.argmin(normalized_signal[s_search_start:s_search_end]) + s_search_start
                if s_peak_idx_cand != -1 and (s_peak_idx_cand > r_idx) and (s_peak_idx_cand - r_idx <= int(0.08 * SAMPLE_RATE)):
                    s_idx_beat = s_peak_idx_cand
                    line += f" S: {int(time_vector[s_idx_beat] * 1000)} ms ✔"
                    valid_counts["S"] += 1
                else:
                    line += f" S: ✘"
            else:
                line += f" S: ✘"
            s_temp.append(s_idx_beat)

            # P-peak (sesuai jendela dan kriteria kode drop Excel)
            p_ref_idx = q_idx_beat if q_idx_beat != -1 else r_idx 
            p_search_start = max(0, p_ref_idx - int(0.2 * SAMPLE_RATE))
            p_search_end = p_ref_idx
            p_idx_beat = -1
            if p_search_start < p_search_end and normalized_signal[p_search_start:p_search_end].size > 0:
                p_peak_idx_cand = np.argmax(normalized_signal[p_search_start:p_search_end]) + p_search_start
                if p_peak_idx_cand != -1 and (p_peak_idx_cand < p_ref_idx) and (p_ref_idx - p_peak_idx_cand <= int(0.2 * SAMPLE_RATE)):
                    p_idx_beat = p_peak_idx_cand
                    line += f" P: {int(time_vector[p_idx_beat] * 1000)} ms ✔"
                    valid_counts["P"] += 1
                else:
                    line += f" P: ✘"
            else:
                line += f" P: ✘"
            p_temp.append(p_idx_beat)

            # T-peak (sesuai jendela dan kriteria kode drop Excel)
            t_ref_idx = s_idx_beat if s_idx_beat != -1 else r_idx 
            t_search_start = t_ref_idx
            t_search_end = min(len(normalized_signal), t_ref_idx + int(0.4 * SAMPLE_RATE))
            t_idx_beat = -1
            if t_search_start < t_search_end and normalized_signal[t_search_start:t_search_end].size > 0:
                t_peak_idx_cand = np.argmax(normalized_signal[t_search_start:t_search_end]) + t_search_start
                if t_peak_idx_cand != -1 and (t_peak_idx_cand > t_ref_idx) and (t_peak_idx_cand - t_ref_idx <= int(0.4 * SAMPLE_RATE)):
                    t_idx_beat = t_peak_idx_cand
                    line += f" T: {int(time_vector[t_idx_beat] * 1000)} ms ✔"
                    valid_counts["T"] += 1
                else:
                    line += f" T: ✘"
            else:
                line += f" T: ✘"
            t_temp.append(t_idx_beat)

            self.result_text.insert(tk.END, line + "\n") #
            r_temp.append(r_idx) 

        # Update global buffers for plotting (hanya yang valid)
        p_peaks_buffer[:] = [idx for idx in p_temp if idx != -1]
        q_peaks_buffer[:] = [idx for idx in q_temp if idx != -1]
        r_peaks_buffer[:] = [idx for idx in r_temp if idx != -1] 
        s_peaks_buffer[:] = [idx for idx in s_temp if idx != -1]
        t_peaks_buffer[:] = [idx for idx in t_temp if idx != -1]

        # --- Hitung dan Tampilkan Akurasi ---
        accuracy_text = "\nAkurasi Deteksi PQRST (per beat):\n"
        total_accuracy_sum = 0
        component_count = 0

        for key in ['P', 'Q', 'R', 'S', 'T']:
            if total_beats > 0:
                acc = (valid_counts[key] / total_beats) * 100
            else:
                acc = 0.0
            
            total_accuracy_sum += acc
            component_count += 1
            mark = "✔" if acc == 100 else "⚠️" if acc >= 70 else "✘"
            accuracy_text += f"{key}: {acc:.1f}% {mark}\n"

        avg_accuracy = total_accuracy_sum / component_count if component_count > 0 else 0
        level = "Baik 👍" if avg_accuracy >= 90 else "Cukup ⚠️" if avg_accuracy >= 70 else "Kurang ❌"
        accuracy_text += f"\nTotal Akurasi PQRST: {avg_accuracy:.1f}% ({level})\n" #
        self.result_text.insert(tk.END, accuracy_text)


    def connect_serial(self):
        global is_running, all_raw_data, SERIAL_PORT, serial_connection
        
        if self.is_connected: 
            messagebox.showinfo("Already Connected", "Sudah terhubung ke port serial.")
            return

        SERIAL_PORT = self.selected_com_port.get()
        print(f"Mencoba terhubung ke {SERIAL_PORT}...")
        
        self.reset_testing() 
        is_running = True 

        self.connect_btn.config(state=tk.DISABLED)
        self.disconnect_btn.config(state=tk.DISABLED)
        self.com_port_dropdown.config(state=tk.DISABLED)

        self.serial_thread = threading.Thread(target=self.read_serial, daemon=True)
        self.serial_thread.start()

    def disconnect_serial(self):
        global is_running, serial_connection
        
        if not self.is_connected: 
            messagebox.showinfo("Not Connected", "Tidak ada koneksi serial yang aktif.")
            return

        print("Memulai proses pemutusan koneksi serial...")
        is_running = False 
        self.is_connected = False 

        if self.serial_thread and self.serial_thread.is_alive():
            self.serial_thread.join(timeout=2.0) 
            if self.serial_thread.is_alive():
                print("Peringatan: Thread serial mungkin tidak berhenti tepat waktu.")
        
        if serial_connection and serial_connection.is_open:
            try:
                serial_connection.close()
                print("Koneksi serial ditutup dari disconnect_serial.")
            except serial.SerialException as e:
                print(f"Error saat menutup serial port: {e}")
                messagebox.showwarning("Close Error", f"Gagal menutup port serial dengan bersih: {e}")
            finally:
                serial_connection = None
        else:
            print("Serial connection sudah tertutup atau tidak pernah terbuka.")
            serial_connection = None

        self.update_connection_status_gui() 
        print("Koneksi serial dihentikan secara manual (GUI updated).")

    def toggle_pause(self):
        global is_paused
        is_paused = not is_paused
        self.pause_btn.config(text='▶️ Resume' if is_paused else '⏸️ Pause')

    def reset_testing(self):
        global ecg_buffer, signals_buffer, p_peaks_buffer, q_peaks_buffer, r_peaks_buffer, s_peaks_buffer, t_peaks_buffer, all_raw_data
        
        with data_lock: 
            ecg_buffer = np.zeros(BUFFER_SIZE) 
        signals_buffer = {}
        p_peaks_buffer[:], q_peaks_buffer[:], r_peaks_buffer[:], s_peaks_buffer[:], t_peaks_buffer[:] = [], [], [], [], []
        
        all_raw_data.clear()

        self.bpm_label.config(text="BPM: N/A")
        self.noise_label.config(text="Noise Reduction: N/A")
        self.result_text.delete('1.0', tk.END) 

        self.fig.clear() 
        self.ax = self.fig.add_subplot(111) 
        titles = ['Original ECG', 'Bandpass Filtered', 'Baseline Corrected',
                    'Wavelet Denoised', 'Smoothed Signal', 'Detected PQRST']
        self.lines = {} 
        for key in titles:
            self.lines[key], = self.ax.plot([], [], label=key, color='black', linewidth=1)
        self.peak_markers = {
            'P': None, 'Q': None, 'R': None, 'S': None, 'T': None
        }
        self.canvas.draw_idle()

    def change_filter(self, event=None):
        self.update_plot(0) 

    def save_image(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.png',
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                                                 title="Save Plot Image")
        if file_path:
            try:
                self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {e}")
                
    def save_result_image(self):
        """
        Menyimpan teks hasil analisis PQRST sebagai gambar, serta plot sinyal PQRST
        dengan marker puncak. Disesuaikan agar sama dengan kode drop Excel.
        """
        if 'Detected PQRST' not in signals_buffer:
            messagebox.showwarning("Warning", "Belum ada sinyal yang diproses.")
            return

        fig_save, ax_save = plt.subplots(figsize=(12, 6))
        sig = signals_buffer['Detected PQRST']
        
        ax_save.plot(time_vector[:len(sig)], sig, color='black', label='ECG')
        self.plot_peak(ax_save, time_vector, sig, p_peaks_buffer, 'm', 'P')
        self.plot_peak(ax_save, time_vector, sig, q_peaks_buffer, 'c', 'Q')
        self.plot_peak(ax_save, time_vector, sig, r_peaks_buffer, 'r', 'R')
        self.plot_peak(ax_save, time_vector, sig, s_peaks_buffer, 'b', 'S')
        self.plot_peak(ax_save, time_vector, sig, t_peaks_buffer, 'g', 'T')
        
        ax_save.set_title('ECG Signal with Detected PQRST')
        ax_save.set_xlabel('Time (s)')
        ax_save.set_ylabel('Amplitude')
        ax_save.grid(True)
        ax_save.legend(loc='upper right')

        info_text = self.result_text.get('1.0', tk.END) #

        ax_save.text(1.02, 0.95, info_text, transform=ax_save.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
                     family='monospace')

        file_path = filedialog.asksaveasfilename(defaultextension='.png',
                                                 filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                                                 title="Save Result Text as Image")
        if file_path:
            try:
                fig_save.tight_layout(rect=[0, 0, 0.85, 1]) 
                fig_save.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Result text saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save result image: {e}")
            finally:
                plt.close(fig_save)

    def save_comparison_excel_realtime(self):
        global all_raw_data

        if not all_raw_data:
            messagebox.showwarning("No Data", "Tidak ada data untuk disimpan. Mulai koneksi serial terlebih dahulu.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                                 filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                                                 title="Save Real-time Data Comparison")
        if file_path:
            try:
                print("Processing all historical data for Excel export. This may take a moment...")
                full_processed_result = self.process_ecg_realtime(np.array(all_raw_data))
                print("Historical data processing complete.")

                smoothed_signal_for_excel = full_processed_result.get('Smoothed Signal', np.full(len(all_raw_data), np.nan))
                
                # Hitung noise reduction untuk keseluruhan data
                noise_reduction_value_str = self.calculate_noise_reduction(np.array(all_raw_data), smoothed_signal_for_excel) #
                
                # Ambil nilai numeriknya dari string "X.YY%"
                if isinstance(noise_reduction_value_str, str) and noise_reduction_value_str.endswith('%'):
                    try:
                        noise_reduction_value_num = float(noise_reduction_value_str.replace('%', '')) #
                    except ValueError:
                        noise_reduction_value_num = np.nan
                else:
                    noise_reduction_value_num = np.nan

                original_power = np.sum(np.array(all_raw_data)**2) #
                smoothed_power = np.sum(smoothed_signal_for_excel**2) #

                # Buat DataFrame hanya dengan dua kolom yang diminta
                df_data = {
                    'Original ECG': np.round(np.array(all_raw_data), 2), #
                    'Pre-processed ECG (Smoothed Signal)': smoothed_signal_for_excel #
                }
                
                df = pd.DataFrame(df_data)
                
                with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='ECG Data', index=False)
                    
                    # Tambahkan ringkasan noise reduction ke sheet 'ECG Data'
                    workbook = writer.book
                    worksheet = writer.sheets['ECG Data']
                    bold_format = workbook.add_format({'bold': True})
                    
                    start_row = len(df) + 2 
                    worksheet.write(start_row, 0, "Ringkasan Noise Reduction Keseluruhan:", bold_format) #
                    worksheet.write(start_row + 1, 0, "Noise Reduction:", bold_format) #
                    worksheet.write(start_row + 1, 1, f"{noise_reduction_value_num:.2f}%") #
                    worksheet.write(start_row + 2, 0, "Original Power:", bold_format) #
                    worksheet.write(start_row + 2, 1, f"{original_power:.2f}") #
                    worksheet.write(start_row + 3, 0, "Processed Power:", bold_format) #
                    worksheet.write(start_row + 3, 1, f"{smoothed_power:.2f}") #
                    
                messagebox.showinfo("Success", f"Data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan data ke Excel: {e}")

    def change_speed(self, val):
        global PLOT_INTERVAL
        try:
            speed_factor = float(val)
            if speed_factor > 0:
                new_interval = int(50 / speed_factor)
                if new_interval < 10: 
                    new_interval = 10 
                PLOT_INTERVAL = new_interval
                self.ani.event_source.interval = PLOT_INTERVAL
            else:
                self.ani.event_source.interval = 50 
        except ValueError:
            pass 

# --- Main execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeECGViewer(root)
    root.protocol("WM_DELETE_WINDOW", app.disconnect_serial) 
    root.protocol("WM_DELETE_WINDOW", app.on_closing)  # Use the new on_closing method
    root.mainloop()