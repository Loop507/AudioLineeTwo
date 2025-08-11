import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import io
import time
import tempfile
import os
import imageio
import subprocess
from scipy.io import wavfile
from datetime import datetime

# Classe AudioVisualizer semplificata
class AudioVisualizer:
    def __init__(self, audio_data, sr, duration=None):
        self.audio_data = audio_data
        self.sr = sr
        self.original_duration = len(audio_data) / sr
        self.duration = min(duration, self.original_duration) if duration else self.original_duration
        self.setup_frequency_analysis()
        
        # Variabili per il tracking dei colori
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
    
    def get_resolution(self, video_quality, aspect_ratio):
        """Determina la risoluzione in pixel per il video"""
        base_resolutions = {
            "Bassa (960x540)": (960, 540),
            "Media (1280x720)": (1280, 720), 
            "Alta (1920x1080)": (1920, 1080)
        }
        
        base_width, base_height = base_resolutions[video_quality]
        
        if aspect_ratio == "16:9 (Standard)":
            return (base_width, base_height)
        elif aspect_ratio == "1:1 (Quadrato)":
            size = min(base_width, base_height)
            return (size, size)
        elif aspect_ratio == "9:16 (Verticale)":
            return (base_height, base_width)
        else:
            return (base_width, base_height)
    
    def get_aspect_ratio_limits(self, aspect_ratio):
        """Determina i limiti degli assi per l'aspect ratio"""
        if aspect_ratio == "16:9 (Standard)":
            return (16, 9)
        elif aspect_ratio == "1:1 (Quadrato)":
            return (10, 10)
        elif aspect_ratio == "9:16 (Verticale)":
            return (9, 16)
        else:
            return (16, 9)
    
    def create_pattern_frame(self, time_idx, pattern_type="waves", colors=None, effects=None,
                            aspect_ratio="16:9 (Standard)", title_settings=None, 
                            resolution_px=None, dpi=100):
        """Crea un frame del pattern basato sulle frequenze - SOLO WAVES con effetti"""
        low_norm, mid_norm, high_norm = self.get_normalized_bands(time_idx)
        
        # Aggiorna statistiche colori
        self.update_color_statistics(low_norm, mid_norm, high_norm)
        
        # Colori default se non specificati
        if colors is None:
            colors = {
                'low': '#FF0000', 'mid': '#0000FF', 'high': '#FFFFFF', 'bg': '#000000'
            }
        
        # Effetti default se non specificati
        if effects is None:
            effects = {
                'intensity': 1.0,
                'speed': 0.1,
                'randomness': 0.0
            }
        
        # Ottieni impostazioni aspect ratio
        xlim, ylim = self.get_aspect_ratio_limits(aspect_ratio)
        
        # Determina figure size basato su risoluzione
        if resolution_px and dpi:
            figsize = (resolution_px[0] / dpi, resolution_px[1] / dpi)
        else:
            # Modalità preview
            base_width = 10
            figsize = (base_width, base_width * (ylim / xlim))
        
        fig, ax = plt.subplots(figsize=figsize, facecolor=colors['bg'], dpi=dpi)
        ax.set_facecolor(colors['bg'])
        
        # Disegna il pattern wave specifico
        if pattern_type == "waves":
            self.draw_classic_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "interference":
            self.draw_interference_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "flowing":
            self.draw_flowing_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "am":
            self.draw_am_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "fm":
            self.draw_fm_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "reflected":
            self.draw_reflected_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
            self.draw_flowing_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
            
        # Aggiungi titolo se specificato
        if title_settings and title_settings['text']:
            self.draw_title(ax, title_settings, xlim, ylim)
            
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.axis('off')
        
        return fig

    def draw_title(self, ax, title_settings, xlim, ylim):
        """Disegna il titolo in base alle impostazioni di posizione"""
        h_pos = title_settings['h_position']
        v_pos = title_settings['v_position']
        
        # Calcola le coordinate in base alla posizione
        if h_pos == "Sinistra":
            x = 0.05 * xlim
            ha = 'left'
        elif h_pos == "Destra":
            x = 0.95 * xlim
            ha = 'right'
        else:  # Centro
            x = 0.5 * xlim
            ha = 'center'
        
        if v_pos == "Sotto":
            y = 0.05 * ylim
            va = 'bottom'
        else:  # Sopra
            y = 0.95 * ylim
            va = 'top'
        
        ax.text(
            x, y, 
            title_settings['text'],
            fontsize=title_settings['fontsize'],
            color=title_settings['color'],
            ha=ha,
            va=va,
            alpha=0.9,
            fontweight='bold'
        )
    
    def draw_classic_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern ondulatorio classico - originale con controlli"""
        x = np.linspace(0, xlim, 500)
        
        # Usa l'indice temporale per sincronizzare le onde con la musica
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        randomness = effects.get('randomness', 0.0)
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            wave = y_offset + low * intensity * np.sin(2 * np.pi * (0.3 + i * 0.2) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['low'], linewidth=4*low*intensity, alpha=0.8)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            wave = y_offset + mid * intensity * 0.8 * np.sin(2 * np.pi * (0.8 + i * 0.4) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid*intensity, alpha=0.7)
        
        # Onde acute - rapide e piccole
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            wave = y_offset + high * intensity * 0.6 * np.sin(2 * np.pi * (1.5 + i * 0.6) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['high'], linewidth=(1.5+high)*intensity, alpha=0.9)
    
    def draw_interference_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern di interferenza strutturato - onde che si incrociano come nell'immagine"""
        x = np.linspace(0, xlim, 1000)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        randomness = effects.get('randomness', 0.0)
        
        # Layer 1: Onde ampie (basse frequenze) - rosse/arancioni
        num_low_waves = int(3 + low * 2)
        for i in range(num_low_waves):
            base_freq = 0.4 + i * 0.3
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            
            # Onda principale
            y1 = ylim/2 + low * intensity * 2.0 * np.sin(2 * np.pi * base_freq * x/xlim + time_offset + random_offset)
            # Onda interferente con fase diversa
            y2 = ylim/2 + low * intensity * 1.5 * np.sin(2 * np.pi * (base_freq * 1.3) * x/xlim - time_offset * 0.7 + random_offset)
            
            ax.plot(x, y1, color=colors['low'], linewidth=3 + low*2*intensity, alpha=0.7)
            ax.plot(x, y2, color=colors['low'], linewidth=2.5 + low*1.5*intensity, alpha=0.5)
        
        # Layer 2: Onde medie (frequenze medie) - blu/turchesi
        num_mid_waves = int(4 + mid * 3)
        for i in range(num_mid_waves):
            base_freq = 1.0 + i * 0.4
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            
            # Pattern di interferenza più complesso
            y1 = ylim/2 + mid * intensity * 1.2 * np.sin(2 * np.pi * base_freq * x/xlim + time_offset * 1.5 + random_offset)
            y2 = ylim/2 + mid * intensity * 0.8 * np.sin(2 * np.pi * (base_freq * 1.6) * x/xlim - time_offset + random_offset)
            y3 = ylim/2 + mid * intensity * 0.6 * np.sin(2 * np.pi * (base_freq * 0.7) * x/xlim + time_offset * 2 + random_offset)
            
            ax.plot(x, y1, color=colors['mid'], linewidth=2 + mid*1.5*intensity, alpha=0.8)
            ax.plot(x, y2, color=colors['mid'], linewidth=1.5 + mid*intensity, alpha=0.6)
            ax.plot(x, y3, color=colors['mid'], linewidth=1 + mid*0.8*intensity, alpha=0.4)
        
        # Layer 3: Onde acute (alte frequenze) - gialle/bianche
        num_high_waves = int(6 + high * 4)
        for i in range(num_high_waves):
            base_freq = 2.0 + i * 0.5
            random_offset = np.random.random() * randomness if randomness > 0 else 0
            
            # Onde rapide e sottili che si intersecano
            y1 = ylim/2 + high * intensity * 0.8 * np.sin(2 * np.pi * base_freq * x/xlim + time_offset * 3 + random_offset)
            y2 = ylim/2 + high * intensity * 0.6 * np.sin(2 * np.pi * (base_freq * 1.2) * x/xlim - time_offset * 2.5 + random_offset)
            y3 = ylim/2 + high * intensity * 0.4 * np.sin(2 * np.pi * (base_freq * 0.8) * x/xlim + time_offset * 4 + random_offset)
            
            ax.plot(x, y1, color=colors['high'], linewidth=1 + high*intensity, alpha=0.9)
            ax.plot(x, y2, color=colors['high'], linewidth=0.8 + high*0.8*intensity, alpha=0.7)
            ax.plot(x, y3, color=colors['high'], linewidth=0.6 + high*0.6*intensity, alpha=0.5)
    
    def draw_flowing_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern completamente nuovo: Onde Stratificate Orizzontali come nell'immagine"""
        x = np.linspace(0, xlim, 1200)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        randomness = effects.get('randomness', 0.0)
        
        # Dividi lo schermo in fasce orizzontali
        num_layers = 12
        layer_height = ylim / num_layers
        
        # Layer inferiori: Onde basse (rosse/arancioni) - lente e ampie
        for layer in range(4):  # Prime 4 fasce dal basso
            y_base = layer * layer_height + layer_height/2
            
            # Multipli onde per layer con frequenze diverse
            for wave_idx in range(3):
                freq = 0.3 + layer * 0.1 + wave_idx * 0.2
                amplitude = low * intensity * (0.4 + 0.3 * (layer/4))
                random_offset = np.random.random() * randomness if randomness > 0 else 0
                
                # Onda principale stratificata
                wave_y = y_base + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset + random_offset)
                
                # Alpha e spessore basati sul layer
                alpha_val = 0.4 + 0.4 * (layer/4)
                line_width = 2 + low * intensity * (1 + layer/4)
                
                ax.plot(x, wave_y, color=colors['low'], linewidth=line_width, alpha=alpha_val)
        
        # Layer centrali: Onde medie (blu/turchesi) - frequenza intermedia
        for layer in range(4, 8):  # Fasce centrali
            y_base = layer * layer_height + layer_height/2
            
            for wave_idx in range(4):  # Più onde per layer
                freq = 0.8 + (layer-4) * 0.2 + wave_idx * 0.3
                amplitude = mid * intensity * (0.3 + 0.2 * ((layer-4)/4))
                random_offset = np.random.random() * randomness if randomness > 0 else 0
                
                wave_y = y_base + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset * 1.5 + random_offset)
                
                alpha_val = 0.5 + 0.3 * ((layer-4)/4)
                line_width = 1.5 + mid * intensity * (0.8 + (layer-4)/4)
                
                ax.plot(x, wave_y, color=colors['mid'], linewidth=line_width, alpha=alpha_val)
        
        # Layer superiori: Onde acute (bianche/gialle) - rapide e sottili
        for layer in range(8, 12):  # Fasce superiori
            y_base = layer * layer_height + layer_height/2
            
            for wave_idx in range(5):  # Molte onde sottili
                freq = 1.5 + (layer-8) * 0.3 + wave_idx * 0.4
                amplitude = high * intensity * (0.2 + 0.15 * ((layer-8)/4))
                random_offset = np.random.random() * randomness if randomness > 0 else 0
                
                wave_y = y_base + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset * 2.5 + random_offset)
                
                alpha_val = 0.6 + 0.4 * ((layer-8)/4)
                line_width = 0.8 + high * intensity * (0.6 + (layer-8)/4)
                
                ax.plot(x, wave_y, color=colors['high'], linewidth=line_width, alpha=alpha_val)
    

    def draw_am_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali con modulazione di ampiezza (AM) per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Low freq - modulazione lenta e ampia
        am_low = (1 + 0.5 * np.sin(2 * np.pi * 0.2 * x/xlim + time_offset)) 
        y_low = ylim*0.3 + low * intensity * am_low * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset)
        ax.plot(x, y_low, color=colors['low'], linewidth=3*low*intensity, alpha=0.8)

        # Mid freq - modulazione media
        am_mid = (1 + 0.4 * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset*1.2))
        y_mid = ylim*0.5 + mid * intensity * am_mid * np.sin(2 * np.pi * 0.8 * x/xlim + time_offset*1.5)
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5*mid*intensity, alpha=0.7)

        # High freq - modulazione veloce
        am_high = (1 + 0.3 * np.sin(2 * np.pi * 0.8 * x/xlim + time_offset*2))
        y_high = ylim*0.7 + high * intensity * am_high * np.sin(2 * np.pi * 1.6 * x/xlim + time_offset*2.2)
        ax.plot(x, y_high, color=colors['high'], linewidth=2*high*intensity, alpha=0.9)

    def draw_fm_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali con modulazione di frequenza (FM) per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Low freq - FM lenta
        freq_low = 0.4 + 0.1 * np.sin(2 * np.pi * 0.2 * x/xlim + time_offset)
        y_low = ylim*0.3 + low * intensity * np.sin(2 * np.pi * freq_low * x/xlim + time_offset)
        ax.plot(x, y_low, color=colors['low'], linewidth=3*low*intensity, alpha=0.8)

        # Mid freq - FM media
        freq_mid = 0.8 + 0.15 * np.sin(2 * np.pi * 0.3 * x/xlim + time_offset*1.3)
        y_mid = ylim*0.5 + mid * intensity * np.sin(2 * np.pi * freq_mid * x/xlim + time_offset*1.4)
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5*mid*intensity, alpha=0.7)

        # High freq - FM veloce
        freq_high = 1.6 + 0.2 * np.sin(2 * np.pi * 0.5 * x/xlim + time_offset*1.8)
        y_high = ylim*0.7 + high * intensity * np.sin(2 * np.pi * freq_high * x/xlim + time_offset*1.9)
        ax.plot(x, y_high, color=colors['high'], linewidth=2*high*intensity, alpha=0.9)

    def draw_reflected_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali riflesse simmetricamente per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Funzione helper per specchiare onde
        def draw_reflected(y_base, amplitude, freq, color, width, alpha):
            y = amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset)
            ax.plot(x, y_base + y, color=color, linewidth=width, alpha=alpha)
            ax.plot(x, y_base - y, color=color, linewidth=width, alpha=alpha)
        
        # Low
        draw_reflected(ylim*0.3, low * intensity, 0.4, colors['low'], 3*low*intensity, 0.8)
        # Mid
        draw_reflected(ylim*0.5, mid * intensity, 0.8, colors['mid'], 2.5*mid*intensity, 0.7)
        # High
        draw_reflected(ylim*0.7, high * intensity, 1.6, colors['high'], 2*high*intensity, 0.9)
    def create_video_no_audio(self, output_path, pattern_type, colors, effects, fps, 
                             aspect_ratio="16:9 (Standard)", video_quality="Media (1280x720)", 
                             title_settings=None):
        """Crea un video senza audio"""
        # Reset statistiche colori
        self.color_statistics = {
            'low_total': 0,
            'mid_total': 0,
            'high_total': 0,
            'total_energy': 0
        }
        
        # Calcola la risoluzione finale
        resolution_px = self.get_resolution(video_quality, aspect_ratio)
        
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
            
            # Crea frame con la risoluzione corretta
            fig = self.create_pattern_frame(
                time_idx, pattern_type, colors, effects, aspect_ratio, 
                title_settings, resolution_px=resolution_px, dpi=100
            )
            
            # Salva il frame come immagine
            frame_path = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
            fig.savefig(frame_path, bbox_inches='tight', pad_inches=0)
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
        
        return total_frames, resolution_px
    
    def create_video_with_audio(self, output_path, pattern_type, colors, effects, fps, 
                               audio_filename="Unknown Track", video_quality="Media (1280x720)", 
                               aspect_ratio="16:9 (Standard)", video_title="My Audio Visual", title_settings=None):
        """Crea un video completo con audio e genera report finale"""
        # Crea un video temporaneo senza audio
        temp_video_path = output_path.replace('.mp4', '_no_audio.mp4')
        total_frames, resolution_px = self.create_video_no_audio(
            temp_video_path, pattern_type, colors, effects, fps, 
            aspect_ratio, video_quality, title_settings
        )
        
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
                                      video_quality, aspect_ratio, title_settings,
                                      resolution_px)
            
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
    
    def show_generation_report(self, audio_filename, video_title, pattern_type, colors, effects, fps, total_frames, video_quality, aspect_ratio, title_settings, resolution_px):
        """Mostra il report dettagliato della generazione"""
        # Calcola le percentuali dei colori
        low_percent, mid_percent, high_percent = self.get_color_percentages()
        
        # Formatta la risoluzione
        final_resolution = f"{resolution_px[0]}x{resolution_px[1]}"
        
        # Mappa nomi pattern
        pattern_names = {
            "waves": "Onde Classiche",
            "interference": "Onde Interferenza Strutturate", 
            "flowing": "Onde Stratificate Orizzontali"
        }
        
        # Determina intensità basata sui moltiplicatori
        intensity_level = effects.get('intensity', 1.0)
        if intensity_level < 0.8:
            intensity_desc = "Bassa"
        elif intensity_level > 1.5:
            intensity_desc = "Alta"
        else:
            intensity_desc = "Media"
        
        # Prepara info titolo
        title_info = "❌ Disabilitato"
        if title_settings and title_settings['text']:
            title_position = f"{title_settings['v_position']} {title_settings['h_position']}"
            title_info = f"{title_settings['text']} ({title_position}, {title_settings['fontsize']}px)"
        
        # Crea il report
        report = f"""
## 📊 Audio Visual Report - WAVES EDITION

**🎬 Video Title:** {video_title}  
**🎵 Audio Track:** {audio_filename}  
**⏱️ Duration:** {self.duration:.1f}s  
**🔊 Sample Rate:** {self.sr:,} Hz  
**📺 Resolution:** {final_resolution}  
**📐 Aspect Ratio:** {aspect_ratio}

### 🌈 Color Distribution by Frequency:
- **🔴 Low Frequencies (20-250Hz):** {low_percent:.1f}%
- **🔵 Mid Frequencies (250-4000Hz):** {mid_percent:.1f}%  
- **⚪ High Frequencies (4000-20000Hz):** {high_percent:.1f}%

### ⚙️ Wave Configuration:
- **🌊 Wave Style:** {pattern_names.get(pattern_type, pattern_type.title())}
- **💪 Intensità:** {intensity_desc} ({intensity_level}x)
- **⚡ Velocità:** {effects.get('speed', 0.1)}x
- **🎲 Casualità:** {effects.get('randomness', 0.0)*100:.0f}%
- **📐 Format:** {aspect_ratio.split(' ')[0]} | **🎬 FPS:** {fps}
- **📝 Title:** {title_info}
- **🖼️ Total Frames:** ~{total_frames:,}

---
*Generated by **AudioLineTwo** - WAVES EDITION BY LOOP507*  
*Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
        """
        
        # Mostra il report in un expander
        with st.expander("📊 **WAVE GENERATION REPORT** - Clicca per vedere i dettagli", expanded=True):
            st.markdown(report)
        
        # Anche come info success
        st.success(f"""
        ✅ **Video Wave generato con successo!**
        
        **Distribuzione Colori:**
        🔴 Basse: {low_percent:.1f}% | 🔵 Medie: {mid_percent:.1f}% | ⚪ Acute: {high_percent:.1f}%
        
        **Dettagli:** {total_frames:,} frames • {fps} FPS • {self.duration:.1f}s • Wave: {pattern_names.get(pattern_type, pattern_type)} • {final_resolution}
        **Effetti:** Intensità {intensity_desc} • Velocità {effects.get('speed', 0.1)}x • Casualità {effects.get('randomness', 0.0)*100:.0f}%
        """)

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo WAVES by Loop507",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.header("🌊 Wave Controls")
    
    # Titolo principale
    st.markdown("# 🌊 AudioLineTwo - WAVES EDITION")
    st.markdown("**BY LOOP507 - Solo Effetti Wave**")
    
    # Upload file audio
    uploaded_file = st.sidebar.file_uploader(
        "Carica file audio",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Formati supportati: WAV, MP3, M4A, FLAC"
    )
    
    # Input titolo video
    video_title = st.sidebar.text_input(
        "Titolo Video", 
        "My Wave Visual", 
        help="Titolo da mostrare nel report"
    )
    
    # Selezione pattern WAVE
    pattern_type = st.sidebar.selectbox(
        "Tipo di Onda",
        ["waves", "interference", "flowing", "am", "fm", "reflected"],
        help="Scegli il tipo di visualizzazione wave",
        format_func=lambda x: {
            "waves": "🌊 Onde Classiche",
            "am": "📡 Onde AM (Ampiezza Modulata)",
            "fm": "🎛️ Onde FM (Frequenza Modulata)",
            "reflected": "🪞 Onde Riflesse",
            "interference": "🔄 Onde Interferenza", 
            "flowing": "💫 Onde Fluide"
        }[x]
    )
    
    # Controlli colori semplificati
    st.sidebar.subheader("🎨 Colori Wave")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        color_low = st.color_picker("Freq. Basse", "#FF0000", help="Colore per frequenze basse (20-250Hz)")
        color_mid = st.color_picker("Freq. Medie", "#0000FF", help="Colore per frequenze medie (250-4000Hz)")
    with col2:
        color_high = st.color_picker("Freq. Acute", "#FFFFFF", help="Colore per frequenze acute (4000-20000Hz)")
        bg_color = st.color_picker("Sfondo", "#000000", help="Colore di sfondo")
    
    # Controlli per il titolo
    st.sidebar.subheader("📝 Impostazioni Titolo")
    title_enabled = st.sidebar.checkbox("Mostra Titolo", value=False)
    title_text = st.sidebar.text_input("Testo Titolo", video_title) if title_enabled else ""
    title_font_size = st.sidebar.slider("Dimensione Font", 10, 50, 20) if title_enabled else 20
    title_color = st.sidebar.color_picker("Colore Titolo", "#FFFFFF") if title_enabled else "#FFFFFF"
    
    if title_enabled:
        # Posizione orizzontale
        title_h_position = st.sidebar.selectbox(
            "Posizione Orizzontale",
            ["Sinistra", "Centro", "Destra"],
            index=1
        )
        
        # Posizione verticale
        title_v_position = st.sidebar.selectbox(
            "Posizione Verticale",
            ["Sopra", "Sotto"],
            index=0
        )
    else:
        title_h_position = "Centro"
        title_v_position = "Sopra"
    
    # Controlli effetti generali
    st.sidebar.subheader("⚙️ Controlli Generali")
    
    # Intensità
    intensity_multiplier = st.sidebar.slider("Intensità", 0.5, 3.0, 1.0, 0.1, 
                                            help="Controlla l'intensità generale degli effetti")
    
    # Velocità
    speed_multiplier = st.sidebar.slider("Velocità", 0.1, 2.0, 0.1, 0.05,
                                       help="Controlla la velocità di movimento delle onde")
    
    # Randomness
    randomness_factor = st.sidebar.slider("Casualità", 0.0, 1.0, 0.0, 0.05,
                                        help="Aggiunge variazione casuale alle onde")
    
    # Preparazione effetti
    effects = {
        'intensity': intensity_multiplier,
        'speed': speed_multiplier,
        'randomness': randomness_factor
    }
    
    # FPS per la visualizzazione
    frame_rate = st.sidebar.selectbox("FPS", [10, 15, 20, 30], index=2)
    
    # Qualità video
    video_quality = st.sidebar.selectbox("Qualità Video", ["Bassa (960x540)", "Media (1280x720)", "Alta (1920x1080)"], index=1)
    
    # Aspect Ratio
    aspect_ratio = st.sidebar.selectbox("Aspect Ratio", ["16:9 (Standard)", "1:1 (Quadrato)", "9:16 (Verticale)"], index=0)
    
    # Prepara impostazioni titolo
    title_settings = {
        'text': title_text if title_enabled else "",
        'fontsize': title_font_size,
        'color': title_color,
        'h_position': title_h_position,
        'v_position': title_v_position
    }
    
    if uploaded_file is not None:
        with st.spinner("🎵 Caricamento e analisi audio..."):
            # Carica file audio
            audio_bytes = uploaded_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            # Calcola la durata effettiva
            duration = len(audio_data) / sr
            
            # Crea visualizzatore con la durata effettiva
            visualizer = AudioVisualizer(audio_data, sr)
            
            # Prepara colori
            colors = {
                'low': color_low,
                'mid': color_mid, 
                'high': color_high,
                'bg': bg_color
            }
            
        st.success(f"✅ Audio caricato! Durata: {duration:.1f}s, Sample Rate: {sr}Hz")
        
        # Controlli playback
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🌊 Avvia Wave Viz"):
                st.session_state.start_viz = True
        
        with col2:
            pattern_names = {
                "waves": "Onde Classiche",
                "interference": "Onde Interferenza", 
                "flowing": "Onde Fluide"
            }
            st.write(f"Wave: **{pattern_names.get(pattern_type, pattern_type)}**")
            
        with col3:
            if st.button("🎥 Crea Video Wave", help="Genera un video della visualizzazione wave"):
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
                fig = visualizer.create_pattern_frame(time_idx, pattern_type, colors, effects, aspect_ratio, title_settings)
                
                # Mostra frame
                placeholder.pyplot(fig, clear_figure=True)
                plt.close(fig)
                
                # Aggiorna progress
                progress = (frame + 1) / total_frames
                progress_bar.progress(progress)
                
                # Controllo timing
                time.sleep(1.0 / frame_rate)
            
            st.session_state.start_viz = False
            st.success("🌊 Visualizzazione wave completata!")
        
        # Creazione video
        if 'create_video' in st.session_state and st.session_state.create_video:
            with st.spinner("🎥 Creazione video wave in corso (potrebbe richiedere alcuni minuti)..."):
                # Crea file video temporaneo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    video_path = tmpfile.name
                
                # Ottieni il nome del file audio per il report
                audio_filename = uploaded_file.name if uploaded_file.name else "Unknown Track"
                
                # Crea il video con audio
                success = visualizer.create_video_with_audio(
                    video_path, pattern_type, colors, effects, frame_rate, 
                    audio_filename, video_quality, aspect_ratio, video_title, title_settings
                )
                
                if success:
                    # Mostra il video e il pulsante di download
                    st.video(video_path)
                    
                    # Pulsante di download
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    pattern_names = {
                        "waves": "classic_waves",
                        "interference": "interference_structured", 
                        "flowing": "stratified_horizontal"
                    }
                    
                    st.download_button(
                        label="📥 Scarica Video Wave",
                        data=video_bytes,
                        file_name=f"audioline_wave_{pattern_names.get(pattern_type, pattern_type)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("Errore nella creazione del video wave.")
                
                # Pulisci lo stato
                st.session_state.create_video = False
    
    else:
        # Schermata iniziale
        st.markdown("""
        ### 🌊 Benvenuto in AudioLineTwo - WAVES EDITION!
        
        Versione semplificata con **solo effetti wave** basati sulle frequenze audio:
        
        - **🔊 Frequenze Basse (20-250Hz)** → Onde ampie e lente
        - **🎸 Frequenze Medie (250-4000Hz)** → Onde di frequenza media  
        - **🎼 Frequenze Acute (4000-20000Hz)** → Onde rapide e sottili
        
        **🌊 Tipi di Wave Disponibili:**
        
        ### 🌊 **Onde Classiche**
        Pattern tradizionale con onde sinusoidali pulite, perfette per visualizzazioni eleganti e rilassanti.
        
        ### 🔄 **Onde Interferenza Strutturate** 
        Onde che si intrecciano in modo più ordinato e strutturato, con layer multipli che creano pattern complessi di interferenza. Perfetto per musica elettronica e dinamica.
        
        ### 💫 **Onde Stratificate Orizzontali**
        Nuovo pattern con onde organizzate in fasce orizzontali, dove ogni banda di frequenza occupa una zona specifica dello schermo. Crea un effetto a strati molto distintivo.
        
        **🎨 Caratteristiche:**
        - **Nessun effetto extra** - solo onde pure
        - **Nessuna griglia** - focus totale sulle wave
        - **3 stili wave unici** basati sulla tua immagine
        - **Colori personalizzabili** per ogni banda di frequenza
        - **Report dettagliato** con distribuzione colori
        - **Aspect ratio multipli:** 16:9, 1:1, 9:16
        - **Qualità video HD** fino a 1920x1080
        
        **Come usare:**
        1. Carica un file audio dalla sidebar
        2. Scegli il tipo di wave (Classiche/Interferenza/Fluide)
        3. Personalizza i colori per ogni banda di frequenza
        4. Configura titolo (opzionale) e qualità video
        5. Avvia la preview o crea direttamente il video!
        
        **🎯 Ottimizzato per:**
        - Performance migliori (meno elementi da renderizzare)
        - Focus sulle onde e frequenze audio
        - Maggiore fluidità nell'animazione
        - Estetica minimalista e pulita
        """)
        
        # Demo wave pattern statico
        st.markdown("### 🌊 Anteprima Stili Wave")
        
        demo_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), facecolor='black')
        x = np.linspace(0, 16, 500)
        
        # Demo onde classiche
        ax1.set_facecolor('black')
        for i in range(3):
            y = 1.5 + i * 0.8 + 0.8 * np.sin(2 * np.pi * (0.5 + i * 0.3) * x/16)
            ax1.plot(x, y, color=['#FF0000', '#0000FF', '#FFFFFF'][i], linewidth=3, alpha=0.8)
        ax1.set_xlim(0, 16)
        ax1.set_ylim(0, 4)
        ax1.set_title('🌊 Onde Classiche', color='white', fontsize=14, pad=20)
        ax1.axis('off')
        
        # Demo interferenza
        ax2.set_facecolor('black')
        for i in range(8):
            freq1 = 0.8 + i * 0.2
            freq2 = 1.0 + i * 0.15
            y1 = 2 + 0.6 * np.sin(2 * np.pi * freq1 * x/16)
            y2 = 2 + 0.4 * np.sin(2 * np.pi * freq2 * x/16 + np.pi/4)
            colors = ['#FF4444', '#4444FF', '#FFFFFF']
            ax2.plot(x, y1, color=colors[i % 3], linewidth=1.5, alpha=0.6)
            ax2.plot(x, y2, color=colors[(i+1) % 3], linewidth=1.5, alpha=0.4)
        ax2.set_xlim(0, 16)
        ax2.set_ylim(0, 4)
        ax2.set_title('🔄 Onde Interferenza', color='white', fontsize=14, pad=20)
        ax2.axis('off')
        
        # Demo fluide
        ax3.set_facecolor('black')
        for i in range(15):
            freq = 1.0 + i * 0.1
            phase = i * np.pi/8
            amplitude = 0.3 + 0.2 * (i/15)
            y = 2 + amplitude * np.sin(2 * np.pi * freq * x/16 + phase)
            
            # Colore sfumato
            color_intensity = 0.3 + 0.7 * (i/15)
            color = (color_intensity, color_intensity * 0.7, color_intensity)
            ax3.plot(x, y, color=color, linewidth=0.8, alpha=0.4 + 0.4*(i/15))
        
        ax3.set_xlim(0, 16)
        ax3.set_ylim(0, 4)
        ax3.set_title('💫 Onde Fluide', color='white', fontsize=14, pad=20)
        ax3.axis('off')
        
        plt.tight_layout()
        st.pyplot(demo_fig, clear_figure=True)
        plt.close(demo_fig)

if __name__ == "__main__":
    main()
