import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import librosa
import io
import time
import tempfile
import os
import imageio
import subprocess
from scipy.signal import find_peaks
from scipy.io import wavfile
from datetime import datetime

# Classe AudioVisualizer migliorata
class AudioVisualizer:
    def __init__(self, audio_data, sr, duration=None):
        self.audio_data = audio_data
        self.sr = sr
        self.original_duration = len(audio_data) / sr
        self.duration = min(duration, self.original_duration) if duration else self.original_duration
        self.setup_frequency_analysis()
        
        # Nuove variabili per il tracking dei colori
        self.color_statistics = {
            'low_total': 0,
            'mid_total': 0,
            'high_total': 0,
            'total_energy': 0
        }
        
    def setup_frequency_analysis(self):
        """Configurazione analisi frequenze"""
        # Parametri per l'analisi FFT
        self.hop_length = 512
        self.n_fft = 2048
        
        # Calcola spettrogramma
        self.stft = librosa.stft(
            self.audio_data, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length
        )
        self.magnitude = np.abs(self.stft)
        
        # Calcola il tempo per ogni frame
        self.times = librosa.times_like(self.stft, sr=self.sr, hop_length=self.hop_length)
        
        # Definisci bande di frequenza
        self.freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.low_freq_idx = np.where((self.freq_bins >= 20) & (self.freq_bins <= 250))[0]
        self.mid_freq_idx = np.where((self.freq_bins >= 250) & (self.freq_bins <= 4000))[0]
        self.high_freq_idx = np.where((self.freq_bins >= 4000) & (self.freq_bins <= 20000))[0]
        
        # Trova il picco massimo per la normalizzazione
        self.max_low = np.max(np.mean(self.magnitude[self.low_freq_idx, :], axis=0))
        self.max_mid = np.max(np.mean(self.magnitude[self.mid_freq_idx, :], axis=0))
        self.max_high = np.max(np.mean(self.magnitude[self.high_freq_idx, :], axis=0))
        
    def get_frequency_bands(self, time_idx):
        """Estrai intensità per bande di frequenza"""
        if time_idx >= self.magnitude.shape[1]:
            return 0, 0, 0
            
        low_energy = np.mean(self.magnitude[self.low_freq_idx, time_idx])
        mid_energy = np.mean(self.magnitude[self.mid_freq_idx, time_idx])
        high_energy = np.mean(self.magnitude[self.high_freq_idx, time_idx])
        
        return low_energy, mid_energy, high_energy
    
    def get_normalized_bands(self, time_idx):
        """Restituisce le bande normalizzate"""
        low, mid, high = self.get_frequency_bands(time_idx)
        
        # Normalizza rispetto ai valori massimi
        low_norm = low / self.max_low if self.max_low > 0 else 0
        mid_norm = mid / self.max_mid if self.max_mid > 0 else 0
        high_norm = high / self.max_high if self.max_high > 0 else 0
        
        # Applica un minimo per evitare valori troppo bassi
        low_norm = max(low_norm, 0.1)
        mid_norm = max(mid_norm, 0.1)
        high_norm = max(high_norm, 0.1)
        
        return low_norm, mid_norm, high_norm
    
    def update_color_statistics(self, low_norm, mid_norm, high_norm):
        """Aggiorna le statistiche sui colori per il calcolo delle percentuali"""
        # Calcola il peso di ogni banda basato sull'energia normalizzata
        total_frame_energy = low_norm + mid_norm + high_norm
        
        self.color_statistics['low_total'] += low_norm
        self.color_statistics['mid_total'] += mid_norm  
        self.color_statistics['high_total'] += high_norm
        self.color_statistics['total_energy'] += total_frame_energy
    
    def get_color_percentages(self):
        """Calcola le percentuali finali di utilizzo dei colori"""
        total = self.color_statistics['total_energy']
        
        if total == 0:
            return 0, 0, 0
            
        low_percent = (self.color_statistics['low_total'] / total) * 100
        mid_percent = (self.color_statistics['mid_total'] / total) * 100
        high_percent = (self.color_statistics['high_total'] / total) * 100
        
        return low_percent, mid_percent, high_percent
    
    def get_aspect_ratio_settings(self, aspect_ratio):
        """Determina le impostazioni per l'aspect ratio"""
        if aspect_ratio == "16:9 (Standard)":
            return (14, 10), (16, 10)
        elif aspect_ratio == "1:1 (Quadrato)":
            return (10, 10), (10, 10)
        elif aspect_ratio == "9:16 (Verticale)":
            return (10, 14), (10, 16)
        else:
            return (14, 10), (16, 10)  # default
    
    def create_pattern_frame(self, time_idx, pattern_type="blocks", colors=None, effects=None, aspect_ratio="16:9 (Standard)"):
        """Crea un frame del pattern basato sulle frequenze"""
        low_norm, mid_norm, high_norm = self.get_normalized_bands(time_idx)
        
        # Aggiorna statistiche colori
        self.update_color_statistics(low_norm, mid_norm, high_norm)
        
        # Colori default se non specificati
        if colors is None:
            colors = {
                'low': '#00BFFF', 'mid': '#00CED1', 'high': '#40E0D0', 'bg': '#1a1a2e'
            }
        
        # Effetti default se non specificati
        if effects is None:
            effects = {
                'size_mult': 1.0, 'movement': 0.5, 'alpha': 0.7, 
                'glow': True, 'grid': True, 'gradient': True
            }
        
        # Ottieni impostazioni aspect ratio
        figsize, (xlim, ylim) = self.get_aspect_ratio_settings(aspect_ratio)
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=colors['bg'])
        ax.set_facecolor(colors['bg'])
        
        if pattern_type == "blocks":
            self.draw_blocks_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "lines":
            self.draw_lines_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "waves":
            self.draw_waves_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "vertical":
            self.draw_vertical_lines_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
            
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.axis('off')
        
        return fig
    
    def draw_blocks_pattern(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern a blocchi colorati"""
        size_mult = effects['size_mult']
        
        # Draw grid if enabled
        if effects['grid']:
            for x in np.arange(0, xlim, 1):
                ax.axvline(x, color='white', alpha=0.1, linewidth=0.3)
            for y in np.arange(0, ylim, 1):
                ax.axhline(y, color='white', alpha=0.1, linewidth=0.3)

        # Blocchi grandi per frequenze basse
        for i in range(int(low * 15)):
            x = np.random.uniform(0, xlim)
            y = np.random.uniform(0, ylim)
            width = np.random.uniform(0.5, 2.0) * low * size_mult
            height = np.random.uniform(0.3, 1.5) * low * size_mult
            color = colors['low']
            alpha = np.clip((0.3 + low * 0.7) * effects['alpha'], 0.0, 1.0)
            
            # Glow effect settings
            glow = effects['glow']
            edgecolor = 'white' if glow else 'none'
            linewidth = 1.0 if glow else 0
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, 
                           edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
        
        # Blocchi medi per frequenze medie
        for i in range(int(mid * 25)):
            x = np.random.uniform(0, xlim)
            y = np.random.uniform(0, ylim)
            width = np.random.uniform(0.2, 1.0) * mid * size_mult
            height = np.random.uniform(0.2, 1.0) * mid * size_mult
            color = colors['mid']
            alpha = np.clip((0.4 + mid * 0.6) * effects['alpha'], 0.0, 1.0)
            
            # Glow effect settings
            glow = effects['glow']
            edgecolor = 'white' if glow else 'none'
            linewidth = 1.0 if glow else 0
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, 
                           edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
        
        # Blocchi piccoli per frequenze alte
        for i in range(int(high * 40)):
            x = np.random.uniform(0, xlim)
            y = np.random.uniform(0, ylim)
            width = np.random.uniform(0.1, 0.5) * high * size_mult
            height = np.random.uniform(0.1, 0.5) * high * size_mult
            color = colors['high']
            alpha = np.clip((0.5 + high * 0.5) * effects['alpha'], 0.0, 1.0)
            
            # Glow effect settings
            glow = effects['glow']
            edgecolor = 'white' if glow else 'none'
            linewidth = 1.0 if glow else 0
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, 
                           edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(rect)
    
    def draw_lines_pattern(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern di linee orizzontali"""
        size_mult = effects['size_mult']
        
        # Draw grid if enabled
        if effects['grid']:
            for x in np.arange(0, xlim, 1):
                ax.axvline(x, color='white', alpha=0.1, linewidth=0.3)
            for y in np.arange(0, ylim, 1):
                ax.axhline(y, color='white', alpha=0.1, linewidth=0.3)

        # Linee spesse per basse
        for i in range(int(low * 8)):
            y_pos = np.random.uniform(1, ylim-1)
            x_start = np.random.uniform(0, xlim*0.25)
            x_end = x_start + np.random.uniform(xlim*0.25, xlim*0.75) * low * size_mult
            x_end = min(x_end, xlim)
            alpha = np.clip(0.7 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_start, x_end], [y_pos, y_pos], 
                       color='white', 
                       linewidth=8*low*size_mult + 2, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_start, x_end], [y_pos, y_pos], 
                   color=colors['low'], linewidth=8*low*size_mult, alpha=alpha)
        
        # Linee medie
        for i in range(int(mid * 12)):
            y_pos = np.random.uniform(1, ylim-1)
            x_start = np.random.uniform(0, xlim*0.375)
            x_end = x_start + np.random.uniform(xlim*0.19, xlim*0.625) * mid * size_mult
            x_end = min(x_end, xlim)
            alpha = np.clip(0.6 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_start, x_end], [y_pos, y_pos], 
                       color='white', 
                       linewidth=4*mid*size_mult + 1.5, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_start, x_end], [y_pos, y_pos], 
                   color=colors['mid'], linewidth=4*mid*size_mult, alpha=alpha)
        
        # Linee sottili per acute
        for i in range(int(high * 20)):
            y_pos = np.random.uniform(1, ylim-1)
            x_start = np.random.uniform(0, xlim*0.5)
            x_end = x_start + np.random.uniform(xlim*0.125, xlim*0.5) * high * size_mult
            x_end = min(x_end, xlim)
            alpha = np.clip(0.8 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_start, x_end], [y_pos, y_pos], 
                       color='white', 
                       linewidth=(1+high)*size_mult + 1, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_start, x_end], [y_pos, y_pos], 
                   color=colors['high'], linewidth=(1+high)*size_mult, alpha=alpha)
    
    def draw_waves_pattern(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern ondulatorio"""
        x = np.linspace(0, xlim, 300)
        size_mult = effects['size_mult']
        
        # Draw grid if enabled
        if effects['grid']:
            for x in np.arange(0, xlim, 1):
                ax.axvline(x, color='white', alpha=0.1, linewidth=0.3)
            for y in np.arange(0, ylim, 1):
                ax.axhline(y, color='white', alpha=0.1, linewidth=0.3)

        # Usa l'indice temporale per sincronizzare le onde con la musica
        time_offset = time_idx * 0.1
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            wave = y_offset + low * np.sin(2 * np.pi * (0.3 + i * 0.2) * x/xlim + time_offset)
            alpha = np.clip(0.8 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot(x, wave, color='white', linewidth=6*low*size_mult + 2, alpha=alpha * 0.4)
            
            ax.plot(x, wave, color=colors['low'], linewidth=6*low*size_mult, alpha=alpha)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            wave = y_offset + mid * 0.8 * np.sin(2 * np.pi * (0.8 + i * 0.4) * x/xlim + time_offset)
            alpha = np.clip(0.7 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot(x, wave, color='white', linewidth=4*mid*size_mult + 1.5, alpha=alpha * 0.4)
            
            ax.plot(x, wave, color=colors['mid'], linewidth=4*mid*size_mult, alpha=alpha)
        
        # Onde acute - rapide e piccole
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            wave = y_offset + high * 0.6 * np.sin(2 * np.pi * (1.5 + i * 0.6) * x/xlim + time_offset)
            alpha = np.clip(0.9 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot(x, wave, color='white', linewidth=(1.5+high)*size_mult + 1, alpha=alpha * 0.4)
            
            ax.plot(x, wave, color=colors['high'], linewidth=(1.5+high)*size_mult, alpha=alpha)
    
    def draw_vertical_lines_pattern(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern: Linee verticali dinamiche - SENZA PALLINI"""
        size_mult = effects['size_mult']
        
        # Draw grid if enabled
        if effects['grid']:
            for x in np.arange(0, xlim, 1):
                ax.axvline(x, color='white', alpha=0.1, linewidth=0.3)
            for y in np.arange(0, ylim, 1):
                ax.axhline(y, color='white', alpha=0.1, linewidth=0.3)

        # Linee spesse per basse frequenze
        for i in range(int(low * 12)):
            x_pos = np.random.uniform(0, xlim)
            height = np.random.uniform(ylim*0.1, ylim*0.8) * low
            y_start = np.random.uniform(0, ylim - height)
            alpha = np.clip(0.7 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                       color='white', 
                       linewidth=6*low*size_mult + 1.5, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                   color=colors['low'], linewidth=6*low*size_mult, alpha=alpha)
        
        # Linee medie per frequenze medie
        for i in range(int(mid * 18)):
            x_pos = np.random.uniform(0, xlim)
            height = np.random.uniform(ylim*0.1, ylim*0.6) * mid
            y_start = np.random.uniform(0, ylim - height)
            alpha = np.clip(0.8 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                       color='white', 
                       linewidth=3*mid*size_mult + 1, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                   color=colors['mid'], linewidth=3*mid*size_mult, alpha=alpha)
        
        # Linee sottili per alte frequenze - SENZA SCATTER/PALLINI
        for i in range(int(high * 25)):
            x_pos = np.random.uniform(0, xlim)
            height = np.random.uniform(ylim*0.05, ylim*0.4) * high
            y_start = np.random.uniform(0, ylim - height)
            alpha = np.clip(0.9 * effects['alpha'], 0.0, 1.0)
            
            # Glow effect: draw white underlay
            if effects['glow']:
                ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                       color='white', 
                       linewidth=(1+high)*size_mult + 0.8, 
                       alpha=alpha * 0.5)
            
            ax.plot([x_pos, x_pos], [y_start, y_start + height], 
                   color=colors['high'], linewidth=(1+high)*size_mult, alpha=alpha)
    
    def create_video_no_audio(self, output_path, pattern_type, colors, effects, fps, aspect_ratio="16:9 (Standard)"):
        """Crea un video senza audio"""
        # Reset statistiche colori
        self.color_statistics = {
            'low_total': 0,
            'mid_total': 0,
            'high_total': 0,
            'total_energy': 0
        }
        
        # Calcola il numero totale di frame
        total_frames = int(self.duration * fps)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Crea una directory temporanea per i frame
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        # Calcola il passo temporale per frame
        time_step = self.times[-1] / total_frames
        
        # Genera tutti i frame
        for frame_idx in range(total_frames):
            # Calcola il tempo corrente
            current_time = frame_idx * time_step
            
            # Trova l'indice temporale più vicino
            time_idx = np.argmin(np.abs(self.times - current_time))
            
            # Crea frame
            fig = self.create_pattern_frame(time_idx, pattern_type, colors, effects, aspect_ratio)
            
            # Salva il frame come immagine
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            frame_paths.append(frame_path)
            
            # Aggiorna progresso
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Generando frame {frame_idx+1}/{total_frames}")
        
        # Crea video dai frame
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
                os.remove(frame_path)  # Rimuovi frame dopo l'uso
        
        # Rimuovi directory temporanea
        os.rmdir(temp_dir)
        status_text.text("✅ Video senza audio creato")
        progress_bar.empty()
        
        return total_frames
    
    def create_video_with_audio(self, output_path, pattern_type, colors, effects, fps, 
                               audio_filename="Unknown Track", video_quality="Media (1280x720)", 
                               aspect_ratio="16:9 (Standard)", video_title="My Audio Visual"):
        """Crea un video completo con audio e genera report finale"""
        # Crea un video temporaneo senza audio
        temp_video_path = output_path.replace('.mp4', '_no_audio.mp4')
        total_frames = self.create_video_no_audio(temp_video_path, pattern_type, colors, effects, fps, aspect_ratio)
        
        # Crea un file audio temporaneo
        temp_audio_path = output_path.replace('.mp4', '.wav')
        
        # Estrai l'audio corrispondente alla durata effettiva
        start_sample = 0
        end_sample = int(self.duration * self.sr)
        audio_segment = self.audio_data[start_sample:end_sample]
        
        # Normalizza e salva l'audio
        if np.max(np.abs(audio_segment)) > 0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
        wavfile.write(temp_audio_path, self.sr, audio_segment)
        
        # Combina video e audio usando FFmpeg
        try:
            command = [
                'ffmpeg',
                '-y',  # Sovrascrivi senza chiedere
                '-i', temp_video_path,
                '-i', temp_audio_path,
                '-c:v', 'copy',  # Copia il video senza ri-encodare
                '-c:a', 'aac',   # Encodare audio in AAC
                '-shortest',     # Termina quando il più corto dei due stream termina
                '-strict', 'experimental',
                output_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Genera e mostra il report finale
            self.show_generation_report(audio_filename, video_title, pattern_type, 
                                      colors, effects, fps, total_frames, 
                                      video_quality, aspect_ratio)
            
            return True
            
        except subprocess.CalledProcessError as e:
            st.error(f"Errore durante la combinazione audio/video: {e.stderr.decode()}")
            return False
        finally:
            # Pulisci i file temporanei
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    def show_generation_report(self, audio_filename, video_title, pattern_type, colors, effects, fps, total_frames, video_quality, aspect_ratio):
        """Mostra il report dettagliato della generazione"""
        # Calcola le percentuali dei colori
        low_percent, mid_percent, high_percent = self.get_color_percentages()
        
        # Determina risoluzione basata su qualità e aspect ratio
        base_resolutions = {
            "Bassa (960x540)": (960, 540),
            "Media (1280x720)": (1280, 720), 
            "Alta (1920x1080)": (1920, 1080)
        }
        
        base_width, base_height = base_resolutions[video_quality]
        
        # Calcola risoluzione finale basata su aspect ratio
        if aspect_ratio == "16:9 (Standard)":
            final_resolution = f"{base_width}x{base_height}"
        elif aspect_ratio == "1:1 (Quadrato)":
            # Usa l'altezza come base per il quadrato
            final_resolution = f"{base_height}x{base_height}"
        elif aspect_ratio == "9:16 (Verticale)":
            # Inverti larghezza e altezza
            final_resolution = f"{base_height}x{base_width}"
        else:
            final_resolution = f"{base_width}x{base_height}"
        
        # Mappa nomi pattern
        pattern_names = {
            "blocks": "Blocchi dinamici",
            "lines": "Linee orizzontali",
            "waves": "Onde sinusoidali",
            "vertical": "Linee verticali"
        }
        
        # Determina intensità basata sui moltiplicatori
        if effects['size_mult'] < 0.8:
            intensity = "Bassa"
        elif effects['size_mult'] > 1.5:
            intensity = "Alta"
        else:
            intensity = "Media"
        
        # Crea il report
        report = f"""
## 📊 Audio & Visual Settings Report

**🎬 Video Title:** {video_title}  
**🎵 Audio Track:** {audio_filename}  
**⏱️ Duration:** {self.duration:.1f}s  
**🔊 Sample Rate:** {self.sr:,} Hz  
**📺 Resolution:** {final_resolution}  
**📐 Aspect Ratio:** {aspect_ratio}
**🎨 Colors are mapped to the average energy of each frequency band, combining algorithmic analysis with a structure inspired by musical perception.**

### 🌈 Color Distribution by Frequency Band:
- **🔴 Low Frequencies (20-250Hz):** {low_percent:.1f}%
- **🔵 Mid Frequencies (250-4000Hz):** {mid_percent:.1f}%  
- **⚪ High Frequencies (4000-20000Hz):** {high_percent:.1f}%

### ⚙️ Visual Configuration:
- **🎭 Style:** {pattern_names.get(pattern_type, pattern_type.title())}
- **🎨 Theme:** Custom  
- **💪 Intensity:** {intensity}
- **📐 Format:** {aspect_ratio.split(' ')[0]} | **🎬 FPS:** {fps}
- **🔊 Volume Offset:** 1.0
- **🖼️ Total Frames:** ~{total_frames:,}

### 🔧 Effects Applied:
- **📏 Size Multiplier:** {effects['size_mult']}x
- **🌊 Movement Speed:** {effects['movement']}
- **🌟 Base Transparency:** {effects['alpha']}
- **✨ Glow Effect:** {'✅ Enabled' if effects['glow'] else '❌ Disabled'}
- **🔲 Grid Mode:** {'✅ Enabled' if effects['grid'] else '❌ Disabled'}
- **🌈 Gradients:** {'✅ Enabled' if effects['gradient'] else '❌ Disabled'}

---
*Generated by **AudioLineTwo** - BY LOOP507*  
*Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
        """
        
        # Mostra il report in un expander
        with st.expander("📊 **GENERATION REPORT** - Clicca per vedere i dettagli", expanded=True):
            st.markdown(report)
        
        # Anche come info success
        st.success(f"""
        ✅ **Video generato con successo!**
        
        **Distribuzione Colori:**
        🔴 Basse: {low_percent:.1f}% | 🔵 Medie: {mid_percent:.1f}% | ⚪ Acute: {high_percent:.1f}%
        
        **Dettagli:** {total_frames:,} frames • {fps} FPS • {self.duration:.1f}s • Pattern: {pattern_names.get(pattern_type, pattern_type)} • {final_resolution}
        """)

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.header("🎛️ Controlli")
    
    # Titolo principale
    st.markdown("# 🎵 AudioLineTwo")
    st.markdown("**BY LOOP507 - Enhanced Version**")
    
    # Upload file audio
    uploaded_file = st.sidebar.file_uploader(
        "Carica file audio",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Formati supportati: WAV, MP3, M4A, FLAC"
    )
    
    # Input titolo video
    video_title = st.sidebar.text_input(
        "Titolo Video", 
        "My Audio Visual", 
        help="Titolo da mostrare nel report"
    )
    
    # Selezione pattern
    pattern_type = st.sidebar.selectbox(
        "Tipo di Pattern",
        ["blocks", "lines", "waves", "vertical"],
        help="Scegli il tipo di visualizzazione"
    )
    
    # Controlli colori personalizzati
    st.sidebar.subheader("🎨 Colori Pattern")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        color_low = st.color_picker("Freq. Basse", "#FF0000", help="Colore per frequenze basse (20-250Hz)")
        color_mid = st.color_picker("Freq. Medie", "#0000FF", help="Colore per frequenze medie (250-4000Hz)")
    with col2:
        color_high = st.color_picker("Freq. Acute", "#FFFFFF", help="Colore per frequenze acute (4000-20000Hz)")
        bg_color = st.color_picker("Sfondo", "#000000", help="Colore di sfondo")
    
    # Controlli effetti
    st.sidebar.subheader("⚙️ Controlli Effetti")
    
    # Dimensioni
    size_multiplier = st.sidebar.slider("Moltiplicatore Dimensione", 0.5, 3.0, 1.0, 0.1)
    
    # Movimento
    movement_speed = st.sidebar.slider("Velocità Movimento", 0.0, 2.0, 0.5, 0.1)
    
    # Trasparenza
    alpha_base = st.sidebar.slider("Trasparenza Base", 0.1, 1.0, 0.7, 0.05)
    
    # Blur/Glow effect
    glow_effect = st.sidebar.checkbox("Effetto Glow", value=True)
    
    # Grid structure
    grid_mode = st.sidebar.checkbox("Modalità Griglia", value=True)
    
    # Sfumature
    gradient_mode = st.sidebar.checkbox("Sfumature", value=True)
    
    # FPS per la visualizzazione
    frame_rate = st.sidebar.selectbox("FPS", [10, 15, 20, 30], index=2)
    
    # Qualità video MODIFICATA
    video_quality = st.sidebar.selectbox("Qualità Video", ["Bassa (960x540)", "Media (1280x720)", "Alta (1920x1080)"], index=1)
    
    # Aspect Ratio personalizzato NUOVO
    aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["16:9 (Standard)", "1:1 (Quadrato)", "9:16 (Verticale)"], index=0)
    
    if uploaded_file is not None:
        with st.spinner("🎵 Caricamento e analisi audio..."):
            # Carica file audio
            audio_bytes = uploaded_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            # Calcola la durata effettiva
            duration = len(audio_data) / sr
            
            # Crea visualizzatore con la durata effettiva
            visualizer = AudioVisualizer(audio_data, sr)
            
            # Prepara colori e effetti
            colors = {
                'low': color_low,
                'mid': color_mid, 
                'high': color_high,
                'bg': bg_color
            }
            
            effects = {
                'size_mult': size_multiplier,
                'movement': movement_speed,
                'alpha': alpha_base,
                'glow': glow_effect,
                'grid': grid_mode,
                'gradient': gradient_mode
            }
            
        st.success(f"✅ Audio caricato! Durata: {duration:.1f}s, Sample Rate: {sr}Hz")
        
        # Controlli playback
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🎬 Avvia Visualizzazione"):
                st.session_state.start_viz = True
        
        with col2:
            st.write(f"Pattern: **{pattern_type.upper()}**")
            
        with col3:
            if st.button("🎥 Crea Video", help="Genera un video della visualizzazione"):
                st.session_state.create_video = True
        
        # Visualizzazione in tempo reale
        if 'start_viz' in st.session_state and st.session_state.start_viz:
            placeholder = st.empty()
            
            total_frames = int(visualizer.duration * frame_rate)
            progress_bar = st.progress(0)
            
            # Calcola il passo temporale per frame
            time_step = visualizer.times[-1] / total_frames
            
            for frame in range(total_frames):
                # Calcola il tempo corrente
                current_time = frame * time_step
                
                # Trova l'indice temporale più vicino
                time_idx = np.argmin(np.abs(visualizer.times - current_time))
                
                # Crea frame
                fig = visualizer.create_pattern_frame(time_idx, pattern_type, colors, effects, aspect_ratio)
                
                # Mostra frame
                placeholder.pyplot(fig, clear_figure=True)
                plt.close(fig)
                
                # Aggiorna progress
                progress = (frame + 1) / total_frames
                progress_bar.progress(progress)
                
                # Controllo timing
                time.sleep(1.0 / frame_rate)
            
            st.session_state.start_viz = False
            st.success("🎉 Visualizzazione completata!")
        
        # Creazione video
        if 'create_video' in st.session_state and st.session_state.create_video:
            with st.spinner("🎥 Creazione video in corso (potrebbe richiedere alcuni minuti)..."):
                # Crea file video temporaneo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    video_path = tmpfile.name
                
                # Ottieni il nome del file audio per il report
                audio_filename = uploaded_file.name if uploaded_file.name else "Unknown Track"
                
                # Crea il video con audio
                success = visualizer.create_video_with_audio(
                    video_path, pattern_type, colors, effects, frame_rate, 
                    audio_filename, video_quality, aspect_ratio, video_title
                )
                
                if success:
                    # Mostra il video e il pulsante di download
                    st.video(video_path)
                    
                    # Pulsante di download
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Scarica Video con Audio",
                        data=video_bytes,
                        file_name=f"audioline_two_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("Errore nella creazione del video. Controlla i log per maggiori dettagli.")
                
                # Pulisci lo stato
                st.session_state.create_video = False
    
    else:
        # Schermata iniziale
        st.markdown("""
        ### 🎵 Benvenuto in AudioLineTwo Enhanced!
        
        Questa applicazione crea pattern visivi dinamici basati sulle frequenze audio:
        
        - **🔊 Frequenze Basse (20-250Hz)** → Pattern grandi e spessi
        - **🎸 Frequenze Medie (250-4000Hz)** → Pattern di dimensione media  
        - **🎼 Frequenze Acute (4000-20000Hz)** → Pattern piccoli e sottili
        
        **🆕 Nuove Funzionalità:**
        - **📊 Calcolo accurato percentuali colori** per ogni banda di frequenza
        - **📋 Report dettagliato finale** con tutte le statistiche del video
        - **🎨 Migliore analisi della distribuzione energetica** 
        - **📐 Aspect Ratio personalizzati:** 16:9, 1:1 (Quadrato), 9:16 (Verticale)
        - **🎯 Qualità video ottimizzata:** 960x540, 1280x720, 1920x1080
        - **🎭 Pattern verticale migliorato** senza elementi di disturbo
        
        **Come usare:**
        1. Carica un file audio dalla sidebar
        2. Scegli il tipo di pattern e personalizza i colori
        3. Seleziona qualità video e aspect ratio desiderati
        4. Configura gli effetti e la qualità
        5. Crea il video per vedere il report completo!
        
        **Pattern disponibili:**
        - **Blocks**: Blocchi rettangolari strutturati
        - **Lines**: Linee orizzontali di spessore variabile
        - **Waves**: Forme ondulatorie dinamiche
        - **Vertical**: Linee verticali pulite (senza pallini)
        
        **📊 Il report finale includerà:**
        - Distribuzione percentuale precisa dei colori utilizzati
        - Risoluzione finale basata su qualità + aspect ratio
        - Statistiche complete dell'audio (durata, sample rate, etc.)
        - Dettagli della configurazione visiva utilizzata
        - Informazioni sui frame generati e impostazioni FPS
        """)
        
        # Demo pattern statico
        st.markdown("### 🎨 Anteprima Pattern")
        
        demo_bg = '#000000'
        demo_fig, demo_ax = plt.subplots(figsize=(14, 8), facecolor=demo_bg)
        demo_ax.set_facecolor(demo_bg)
        
        # Crea demo semplice senza AudioVisualizer
        colors = ['#FF0000', '#0000FF', '#FFFFFF']
        
        # Griglia demo 6x8
        for row in range(6):
            for col in range(8):
                x = col * 1.8 + 0.5
                y = row * 1.3 + 0.5
                
                # Scegli colore basato sulla posizione
                color_idx = (col + row) % 3
                color = colors[color_idx]
                
                # Varia dimensioni
                if color_idx == 0:  # Basse - grandi
                    width, height = 1.6, 1.1
                elif color_idx == 1:  # Medie
                    width, height = 1.3, 0.9
                else:  # Acute - piccoli
                    width, height = 1.0, 0.7
                
                # Alpha variabile
                alpha = 0.4 + (color_idx + 1) * 0.2
                
                # Crea rettangolo
                rect = patches.Rectangle((x, y), width, height,
                                       facecolor=color, alpha=alpha,
                                       edgecolor='white', linewidth=0.3)
                demo_ax.add_patch(rect)
        
        demo_ax.set_xlim(0, 16)
        demo_ax.set_ylim(0, 10)
        demo_ax.axis('off')
        
        st.pyplot(demo_fig, clear_figure=True)
        plt.close(demo_fig)

if __name__ == "__main__":
    main()
