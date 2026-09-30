import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import pywt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variable
fs = 100    # Sampling frequency (100 Hz)
nyquist = fs / 2

ecg_signal_raw = np.zeros(1000) # Menyimpan sinyal raw
ecg_signal_processed = np.zeros(1000) # Menyimpan sinyal setelah pre-processing
processed_signals = {} # Menyimpan semua sinyal setelah pre-processing
time = np.arange(len(ecg_signal_raw)) / fs
signals = {}
p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = [], [], [], [], []
confidence_scores = {"P": [], "Q": [], "R": [], "S": [], "T": []}
noise_reduction_percentage = "N/A"    # Inisialisasi variabel pengurangan noise
bpm_value = "N/A"    # Inisialisasi variabel BPM

# GUI setup
root = tk.Tk()
root.title("ECG Viewer")
root.geometry('1300x950')
root.configure(bg='#e6f2ff')

# --- Dropdown and canvas ---
titles = ['Original ECG', 'Bandpass Filtered', 'Baseline Corrected',
          'Wavelet Denoised', 'Smoothed Signal', 'Detected PQRST', 'All Signals']
selected = tk.StringVar(value='Detected PQRST')
dropdown = ttk.Combobox(root, textvariable=selected, values=titles,
                         font=('Arial', 12), state='readonly')
dropdown.pack(pady=10)

# Label untuk menampilkan BPM
bpm_label = tk.Label(root, text="BPM: N/A", font=('Arial', 14), bg='#e6f2ff')
bpm_label.pack(pady=5)

# Label untuk menampilkan persentase pengurangan noise
noise_label = tk.Label(root, text="Noise Reduction: N/A", font=('Arial', 14), bg='#e6f2ff')
noise_label.pack(pady=5)

fig, ax = plt.subplots(figsize=(12, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Text output untuk hasil analisis PQRST
result_text = tk.Text(root, height=12, font=('Courier New', 10), bg='white')
result_text.pack(fill=tk.X, padx=10, pady=5)

def ensure_positive_and_round(signal_data, original_data):
    min_original = np.min(original_data)
    epsilon = 1e-9
    processed_signal_data = signal_data.copy()
    if min_original < 0:
        processed_signal_data += abs(min_original) + epsilon
    elif np.any(processed_signal_data <= 0): # Handle kasus jika min original 0 tapi ada nilai <= 0
        processed_signal_data += epsilon
    return np.round(processed_signal_data, 2)

def calculate_bpm(r_peak_indices, sampling_rate):
    if len(r_peak_indices) < 2:
        return "N/A"
    intervals = np.diff(r_peak_indices) / sampling_rate    # Selisih waktu antar R-R dalam detik
    if np.any(intervals <= 0):
        return "Error: Non-positive interval"
    mean_interval = np.mean(intervals)
    bpm = 60 / mean_interval
    return f"{bpm:.2f}"

def process_ecg(signal_raw):
    global time, signals, p_peaks, q_peaks, r_peaks, s_peaks, t_peaks, confidence_scores, noise_reduction_percentage, bpm_value, ecg_signal_processed, processed_signals
    time = np.arange(len(signal_raw)) / fs
    confidence_scores = {"P": [], "Q": [], "R": [], "S": [], "T": []}
    p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = [], [], [], [], [] # Reset peaks for new signal

    # Bandpass filter
    b, a = signal.butter(6, [0.5 / nyquist, 40 / nyquist], btype='band')
    filtered = signal.filtfilt(b, a, signal_raw)
    filtered_positive = ensure_positive_and_round(filtered, signal_raw)
    processed_signals['Bandpass Filtered'] = filtered_positive

    # Baseline correction
    baseline = gaussian_filter1d(filtered, sigma=fs * 0.3 / 6)
    corrected = filtered - baseline
    corrected_positive = ensure_positive_and_round(corrected, signal_raw)
    processed_signals['Baseline Corrected'] = corrected_positive

    # Wavelet denoising
    wavelet = 'coif4'
    coeffs = pywt.wavedec(corrected, wavelet, level=5)
    thresholded = [coeffs[0]] + [pywt.threshold(c, np.median(np.abs(c)) / 0.6745, 'soft') for c in coeffs[1:]]
    denoised = pywt.waverec(thresholded, wavelet)[:len(corrected)]

    # MODIFIKASI: Kurangi nilai setelah denoising sebesar 6
    denoised_modified = denoised - 6
    denoised_positive = ensure_positive_and_round(denoised_modified, signal_raw)
    processed_signals['Wavelet Denoised'] = denoised_positive

    # Smoothing
    smoothed = signal.convolve(denoised_modified, signal.windows.bartlett(int(fs * 0.05)), mode='same')
    smoothed_positive = ensure_positive_and_round(smoothed, signal_raw)
    processed_signals['Smoothed Signal'] = smoothed_positive
    ecg_signal_processed = smoothed_positive # Simpan sinyal yang telah di-smoothing untuk perbandingan noise

    # Normalization
    normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)
    normalized_positive = ensure_positive_and_round(normalized, signal_raw)
    processed_signals['Detected PQRST'] = normalized_positive

    # Calculate Noise Reduction
    original_power = np.sum(signal_raw**2)
    smoothed_power = np.sum(smoothed_positive**2)
    if original_power > 0:
        reduction = 100 * (1 - (smoothed_power / original_power))
        noise_reduction_percentage = f"{reduction:.2f}%"
    else:
        noise_reduction_percentage = "0.00%"


    # R-peak detection
    r_peaks_temp, _ = signal.find_peaks(normalized, height=0.5, distance=int(0.5 * fs))
    r_peaks.extend(r_peaks_temp)
    bpm_value = calculate_bpm(r_peaks, fs)

    # Q, S, P, T peak detection
    for r in r_peaks:
        q_search_start = max(0, r - int(0.1 * fs))
        q_search_end = r
        if q_search_start < q_search_end:
            q_peak_idx = np.argmin(normalized[q_search_start:q_search_end]) + q_search_start
            if q_peak_idx < len(normalized):
                q_peaks.append(q_peak_idx)
            else:
                q_peaks.append(-1)
        else:
            q_peaks.append(-1)

        s_search_start = r
        s_search_end = min(len(normalized), r + int(0.1 * fs))
        if s_search_start < s_search_end:
            s_peak_idx = np.argmin(normalized[s_search_start:s_search_end]) + s_search_start
            if s_peak_idx < len(normalized):
                s_peaks.append(s_peak_idx)
            else:
                s_peaks.append(-1)
        else:
            s_peaks.append(-1)

        p_search_start = max(0, q_search_start - int(0.2 * fs))
        p_search_end = q_search_start
        if p_search_start < p_search_end:
            p_peak_idx = np.argmax(normalized[p_search_start:p_search_end]) + p_search_start
            if p_peak_idx < len(normalized):
                p_peaks.append(p_peak_idx)
            else:
                p_peaks.append(-1)
        else:
            p_peaks.append(-1)

        t_search_start = s_search_end
        t_search_end = min(len(normalized), s_search_end + int(0.4 * fs))
        if t_search_start < t_search_end:
            t_peak_idx = np.argmax(normalized[t_search_start:t_search_end]) + t_search_start
            if t_peak_idx < len(normalized):
                t_peaks.append(t_peak_idx)
            else:
                t_peaks.append(-1)
        else:
            t_peaks.append(-1)

    # Simpan semua sinyal untuk plotting
    signals = {
        'Original ECG': signal_raw,
        'Bandpass Filtered': filtered_positive,
        'Baseline Corrected': corrected_positive,
        'Wavelet Denoised': denoised_positive,
        'Smoothed Signal': smoothed_positive,
        'Detected PQRST': normalized_positive
    }

    # Tampilkan hasil analisis PQRST
    result_text.delete('1.0', tk.END)
    total = len(r_peaks)
    valid_counts = {"P": 0, "Q": 0, "R": total, "S": 0, "T": 0}

    for i in range(total):
        r = r_peaks[i]
        q = q_peaks[i] if i < len(q_peaks) else -1
        s = s_peaks[i] if i < len(s_peaks) else -1
        p = p_peaks[i] if i < len(p_peaks) else -1
        t = t_peaks[i] if i < len(t_peaks) else -1

        line = f"Beat {i + 1}:"
        line += f"     R: {int(time[r]*1000)} ms ✔"

        if q == -1 or q > r or r - q > int(0.08 * fs):
            line += f"     Q: ✘"
        else:
            line += f"     Q: {int(time[q]*1000)} ms ✔"
            valid_counts["Q"] += 1

        if s == -1 or s < r or s - r > int(0.08 * fs):
            line += f"     S: ✘"
        else:
            line += f"     S: {int(time[s]*1000)} ms ✔"
            valid_counts["S"] += 1

        if p == -1 or p > q or q - p > int(0.2 * fs):
            line += f"     P: ✘"
        else:
            line += f"     P: {int(time[p]*1000)} ms ✔"
            valid_counts["P"] += 1

        if t == -1 or t < s or t - s > int(0.4 * fs):
            line += f"     T: ✘"
        else:
            line += f"     T: {int(time[t]*1000)} ms ✔"
            valid_counts["T"] += 1

        result_text.insert(tk.END, line + "\n")

    result_text.insert(tk.END, "\nAkurasi PQRST:\n")
    total_accuracy = 0
    for key in ['P', 'Q', 'R', 'S', 'T']:
        acc = (valid_counts[key] / total * 100) if total > 0 else 0
        total_accuracy += acc
        mark = "✔" if acc == 100 else "⚠️" if acc >= 70 else "✘"
        result_text.insert(tk.END, f"{key}: {acc:.1f}% {mark}\n")

    avg_accuracy = total_accuracy / 5
    if avg_accuracy >= 90:
        level = "Baik 👍"
    elif avg_accuracy >= 70:
        level = "Cukup ⚠️"
    else:
        level = "Kurang ❌"

    result_text.insert(tk.END, f"\nTotal Akurasi PQRST: {avg_accuracy:.1f}% ({level})\n")


def load_excel():
    global ecg_signal_raw
    file_path = filedialog.askopenfilename(filetypes=[('Excel Files', '*.xlsx')])
    if not file_path:
        return
    try:
        df = pd.read_excel(file_path, skiprows=1, names=['sample', 'ecg'])
        ecg_signal_raw = df['ecg'].values
        process_ecg(ecg_signal_raw.copy()) # Kirim salinan agar data raw tidak terubah langsung
        reset_view()
        messagebox.showinfo("Success", "File loaded and processed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Gagal memproses file:\n{e}")

current_idx = 0
win_size = 150
speed = 1.0
is_paused = False

def update(frame):
    global current_idx, ax, noise_label, bpm_label
    if not is_paused and len(signals) > 0:
        current_idx += int(speed)
        if current_idx > len(ecg_signal_raw) - win_size:
            current_idx = 0

    choice = selected.get()
    fig.clear()

    if choice == 'All Signals' and signals:
        axs = fig.subplots(3, 2).flatten()
        for i, (name, sig) in enumerate(signals.items()):
            start, end = current_idx, current_idx + win_size
            axs[i].plot(time[start:end], sig[start:end])
            axs[i].set_title(name)
            axs[i].grid(True)
            axs[i].set_xlim(time[start], time[end - 1])
            if name == 'Detected PQRST':
                axs[i].plot(time[p_peaks], sig[p_peaks], 'mo', label='P')
                axs[i].plot(time[q_peaks], sig[q_peaks], 'co', label='Q')
                axs[i].plot(time[r_peaks], sig[r_peaks], 'ro', label='R')
                axs[i].plot(time[s_peaks], sig[s_peaks], 'bo', label='S')
                axs[i].plot(time[t_peaks], sig[t_peaks], 'go', label='T')
                axs[i].legend()
        fig.tight_layout()
    elif choice in signals:
        ax = fig.add_subplot(111)
        sig = signals[choice]
        start, end = current_idx, current_idx + win_size
        ax.plot(time[start:end], sig[start:end], color='black')
        ax.set_title(choice)
        ax.grid(True)
        ax.set_xlim(time[start], time[end - 1])
        if choice == 'Detected PQRST':
            ax.plot(time[p_peaks], sig[p_peaks], 'mo', label='P')
            ax.plot(time[q_peaks], sig[q_peaks], 'co', label='Q')
            ax.plot(time[r_peaks], sig[r_peaks], 'ro', label='R')
            ax.plot(time[s_peaks], sig[s_peaks], 'bo', label='S')
            ax.plot(time[t_peaks], sig[t_peaks], 'go', label='T')
            ax.legend()
    canvas.draw()

    # Update label pengurangan noise dan BPM
    noise_label.config(text=f"Noise Reduction: {noise_reduction_percentage}")
    bpm_label.config(text=f"BPM: {bpm_value}")

def toggle_pause():
    global is_paused
    is_paused = not is_paused
    pause_btn.config(text='▶️ Resume' if is_paused else '⏸️ Pause')
    
def reset_view():
    global current_idx
    current_idx = 0

def change_speed(val):
    global speed
    speed = float(val)

def save_image():
    if not signals:
        messagebox.showwarning("Warning", "Tidak ada sinyal yang diproses untuk disimpan.")
        return

    file = filedialog.asksaveasfilename(defaultextension='.png',
                                         filetypes=[('PNG', '*.png')])
    if file:
        fig_save, axs_save = plt.subplots(3, 2, figsize=(15, 10))
        axs_flat = axs_save.flatten()
        plot_titles = list(signals.keys())

        for i, title in enumerate(plot_titles[:6]):
            ax = axs_flat[i]
            sig = signals[title]
            ax.plot(time, sig, color='black')
            ax.set_title(title)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.grid(True)

            if title == 'Detected PQRST':
                ax.plot(time[p_peaks], sig[p_peaks], 'mo', label='P')
                ax.plot(time[q_peaks], sig[q_peaks], 'co', label='Q')
                ax.plot(time[r_peaks], sig[r_peaks], 'ro', label='R')
                ax.plot(time[s_peaks], sig[s_peaks], 'bo', label='S')
                ax.plot(time[t_peaks], sig[t_peaks], 'go', label='T')
                ax.legend()

        fig_save.suptitle(f"All Processed ECG Signals\nBPM: {bpm_value}, Noise Reduction: {noise_reduction_percentage}", fontsize=16)
        fig_save.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap with suptitle
        fig_save.savefig(file, dpi=300)
        plt.close(fig_save)
        messagebox.showinfo("Saved", f"Enam plot sinyal disimpan ke:\n{file}")

def save_result_image():
    if 'Detected PQRST' not in signals:
        messagebox.showwarning("Warning", "Belum ada sinyal yang diproses.")
        return

    fig_save, ax_save = plt.subplots(figsize=(12, 6))
    sig = signals['Detected PQRST']
    ax_save.plot(time, sig, color='black', label='ECG')
    ax_save.plot(time[p_peaks], sig[p_peaks], 'mo', label='P')
    ax_save.plot(time[q_peaks], sig[q_peaks], 'co', label='Q')
    ax_save.plot(time[r_peaks], sig[r_peaks], 'ro', label='R')
    ax_save.plot(time[s_peaks], sig[s_peaks], 'bo', label='S')
    ax_save.plot(time[t_peaks], sig[t_peaks], 'go', label='T')
    ax_save.set_title('ECG Signal with Detected PQRST')
    ax_save.set_xlabel('Time (s)')
    ax_save.set_ylabel('Amplitude')
    ax_save.grid(True)
    ax_save.legend(loc='upper right')

    # Gunakan variabel global dan logika perhitungan akurasi yang sama dari process_ecg
    total = len(r_peaks)
    info_text = f"BPM: {bpm_value}\nNoise Reduction: {noise_reduction_percentage}\n\nAkurasi PQRST (per beat):\n"
    valid_counts = {"P": 0, "Q": 0, "R": total, "S": 0, "T": 0}

    for i in range(total):
        r_time_ms = int(time[r_peaks[i]] * 1000) if 0 <= r_peaks[i] < len(time) else "N/A"
        line = f"Beat {i + 1}: R: {r_time_ms} ms ✔"

        q_valid = 0 <= q_peaks[i] < len(sig) if i < len(q_peaks) and q_peaks[i] != -1 else False
        if q_valid and (q_peaks[i] == -1 or q_peaks[i] > r_peaks[i] or r_peaks[i] - q_peaks[i] > int(0.08 * fs)):
            line += f" Q: ✘"
        elif q_valid:
            line += f" Q: {int(time[q_peaks[i]] * 1000)} ms ✔"
            valid_counts["Q"] += 1
        else:
            line += f" Q: ✘"

        s_valid = 0 <= s_peaks[i] < len(sig) if i < len(s_peaks) and s_peaks[i] != -1 else False
        if s_valid and (s_peaks[i] == -1 or s_peaks[i] < r_peaks[i] or s_peaks[i] - r_peaks[i] > int(0.08 * fs)):
            line += f" S: ✘"
        elif s_valid:
            line += f" S: {int(time[s_peaks[i]] * 1000)} ms ✔"
            valid_counts["S"] += 1
        else:
            line += f" S: ✘"

        p_valid = 0 <= p_peaks[i] < len(sig) if i < len(p_peaks) and p_peaks[i] != -1 else False
        if p_valid and (p_peaks[i] == -1 or p_peaks[i] > q_peaks[i] or q_peaks[i] - p_peaks[i] > int(0.2 * fs)):
            line += f" P: ✘"
        elif p_valid:
            line += f" P: {int(time[p_peaks[i]] * 1000)} ms ✔"
            valid_counts["P"] += 1
        else:
            line += f" P: ✘"

        t_valid = 0 <= t_peaks[i] < len(sig) if i < len(t_peaks) and t_peaks[i] != -1 else False
        if t_valid and (t_peaks[i] == -1 or t_peaks[i] < s_peaks[i] or t_peaks[i] - s_peaks[i] > int(0.4 * fs)):
            line += f" T: ✘"
        elif t_valid:
            line += f" T: {int(time[t_peaks[i]] * 1000)} ms ✔"
            valid_counts["T"] += 1
        else:
            line += f" T: ✘"

        info_text += line + "\n"

    info_text += "\nAkurasi PQRST (keseluruhan):\n"
    total_accuracy = 0
    for key in ['P', 'Q', 'R', 'S', 'T']:
        acc = (valid_counts[key] / total * 100) if total > 0 else 0
        total_accuracy += acc
        mark = "✔" if acc == 100 else "⚠️" if acc >= 70 else "✘"
        info_text += f"{key}: {acc:.1f}% {mark}\n"

    avg_accuracy = total_accuracy / 5
    if avg_accuracy >= 90:
        level = "Baik 👍"
    elif avg_accuracy >= 70:
        level = "Cukup ⚠️"
    else:
        level = "Kurang ❌"

    info_text += f"\nTotal Akurasi PQRST: {avg_accuracy:.1f}% ({level})\n"

    # Tambahkan teks informasi ke plot
    ax_save.text(1.02, 0.95, info_text, transform=ax_save.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    file = filedialog.asksaveasfilename(defaultextension='.png',
                                         filetypes=[('PNG', '*.png')])
    if file:
        fig_save.tight_layout(rect=[0, 0, 0.85, 1])
        fig_save.savefig(file, dpi=300)
        plt.close(fig_save)
        messagebox.showinfo("Saved", f"Hasil keseluruhan disimpan ke:\n{file}")

def save_comparison_excel():
    global ecg_signal_raw, ecg_signal_processed
    if len(ecg_signal_raw) == 0:
        messagebox.showwarning("Warning", "Tidak ada data ECG yang dimuat.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension='.xlsx',
                                             filetypes=[('Excel Files', '*.xlsx')])
    if file_path:
        # Bulatkan juga sinyal raw saat disimpan untuk perbandingan yang lebih baik
        rounded_ecg_raw = np.round(ecg_signal_raw, 2)
        
        # Hitung original_power dan smoothed_power untuk informasi di Excel
        original_power = np.sum(ecg_signal_raw**2)
        smoothed_power = np.sum(ecg_signal_processed**2)
        
        noise_reduction_value = 0
        if original_power > 0:
            noise_reduction_value = 100 * (1 - (smoothed_power / original_power))

        df_comparison = pd.DataFrame({
            'Original ECG': rounded_ecg_raw,
            'Pre-processed ECG (Smoothed Signal)': ecg_signal_processed 
        })
        
        try:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                # Tulis DataFrame utama ke sheet pertama
                df_comparison.to_excel(writer, sheet_name='ECG Data', index=False)
                
                # Dapatkan objek workbook dan worksheet dari writer
                workbook = writer.book
                worksheet = writer.sheets['ECG Data']

                # Definisikan format untuk teks
                bold_format = workbook.add_format({'bold': True})
                
                # Tulis ringkasan di bawah data, dimulai dari kolom A
                # Tambahkan beberapa baris kosong setelah data untuk estetika
                start_row = len(df_comparison) + 2 
                
                worksheet.write(start_row, 0, "Ringkasan Noise Reduction Keseluruhan:", bold_format)
                worksheet.write(start_row + 1, 0, "Noise Reduction:", bold_format)
                worksheet.write(start_row + 1, 1, f"{noise_reduction_value:.2f}%")
                worksheet.write(start_row + 2, 0, "Original Power:", bold_format)
                worksheet.write(start_row + 2, 1, f"{original_power:.2f}") # Dibulatkan dua desimal
                worksheet.write(start_row + 3, 0, "Processed Power:", bold_format)
                worksheet.write(start_row + 3, 1, f"{smoothed_power:.2f}") # Dibulatkan dua desimal
                
            messagebox.showinfo("Saved", f"Data perbandingan disimpan ke:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menyimpan ke Excel:\n{e}")

ctrl = tk.Frame(root, bg='#e6f2ff')
ctrl.pack(pady=10)

load_btn = tk.Button(ctrl, text='📂 Drop File Excel', command=load_excel,
                     font=('Arial', 12), bg='#6c63ff', fg='white')
load_btn.grid(row=0, column=0, padx=5)

pause_btn = tk.Button(ctrl, text='⏸️ Pause', command=toggle_pause,
                     font=('Arial', 12), bg='#4CAF50', fg='white')
pause_btn.grid(row=0, column=1, padx=5)

reset_btn = tk.Button(ctrl, text='🔄 Reset', command=reset_view,
                     font=('Arial', 12), bg='#2196F3', fg='white')
reset_btn.grid(row=0, column=2, padx=5)

save_btn = tk.Button(ctrl, text='💾 Save Image', command=save_image,
                     font=('Arial', 12), bg='#f57c00', fg='white')
save_btn.grid(row=0, column=3, padx=5)

speed_slider = tk.Scale(ctrl, from_=0.5, to=3.0, resolution=0.1,
                         orient=tk.HORIZONTAL, label='Speed',
                         command=change_speed, bg='#e6f2ff')
speed_slider.set(1.0)
speed_slider.grid(row=0, column=4, padx=5)

save_result_btn = tk.Button(ctrl, text='📸 Save Result', command=save_result_image,
                             font=('Arial', 12), bg='#9C27B0', fg='white')
save_result_btn.grid(row=0, column=5, padx=5)

save_comparison_btn = tk.Button(ctrl, text='📊 Save Comparison', command=save_comparison_excel,
                                 font=('Arial', 12), bg='#008080', fg='white')
save_comparison_btn.grid(row=0, column=6, padx=5)

dropdown.bind('<<ComboboxSelected>>', lambda e: reset_view())
ani = FuncAnimation(fig, update, interval=50)
update(0)

root.mainloop()