import streamlit as st
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import librosa
import io
import hashlib
import tempfile
import os
import shutil
import subprocess
import imageio_ffmpeg
from scipy.io import wavfile
from datetime import datetime

# Riferimento temporale: ~86.13 "tick" al secondo (44100 Hz / hop 512).
# La velocità delle onde dipende dai secondi, non dal sample rate del file.
REF_TICKS_PER_SEC = 44100 / 512
BAND_FLOOR = 0.1

# Classe AudioVisualizer semplificata
class AudioVisualizer:
    """Analizzatore + renderer. Dopo __init__ non muta stato: sicuro con st.cache_resource."""

    def __init__(self, audio_data, sr, duration=None):
        self.audio_data = audio_data
        self.sr = sr
        self.original_duration = len(audio_data) / sr
        self.duration = min(duration, self.original_duration) if duration else self.original_duration
        self.setup_frequency_analysis()

    @staticmethod
    def _band_idx(freqs, lo, hi, last=False):
        """Indici dei bin in [lo, hi) (l'ultima banda include hi). Mai vuoto."""
        mask = (freqs >= lo) & ((freqs <= hi) if last else (freqs < hi))
        idx = np.where(mask)[0]
        if idx.size == 0:
            idx = np.array([int(np.argmin(np.abs(freqs - lo)))])
        return idx

    def setup_frequency_analysis(self):
        """Analisi FFT: calcola una volta sola gli inviluppi delle 3 bande, poi libera lo spettrogramma"""
        self.hop_length = 512
        self.n_fft = 2048

        stft = librosa.stft(self.audio_data, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        del stft

        self.times = librosa.times_like(magnitude, sr=self.sr, hop_length=self.hop_length)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        bands = [
            self._band_idx(freqs, 20, 250),
            self._band_idx(freqs, 250, 4000),
            self._band_idx(freqs, 4000, 20000, last=True),
        ]

        # Inviluppo medio per banda (per il disegno) e potenza reale (per il report)
        env = np.vstack([magnitude[idx, :].mean(axis=0) for idx in bands])
        self.band_power = np.vstack([(magnitude[idx, :] ** 2).sum(axis=0) for idx in bands])
        del magnitude  # evita centinaia di MB su brani lunghi

        peak = env.max(axis=1, keepdims=True)
        norm = np.divide(env, peak, out=np.zeros_like(env), where=peak > 0)
        self.norm_bands = np.maximum(norm, BAND_FLOOR)

    def get_normalized_bands(self, time_idx):
        """Restituisce le bande normalizzate (0.1 - 1.0)"""
        t = min(max(int(time_idx), 0), self.norm_bands.shape[1] - 1)
        low, mid, high = self.norm_bands[:, t]
        return float(low), float(mid), float(high)

    def _time_offset(self, time_idx, effects):
        """Fase temporale in secondi (indipendente dal sample rate)"""
        t = float(self.times[min(max(int(time_idx), 0), len(self.times) - 1)])
        return t * REF_TICKS_PER_SEC * effects.get('speed', 0.1)

    @staticmethod
    def _rand(effects):
        """Offset casuale riproducibile: l'rng è seedato per (seed, frame)"""
        randomness = effects.get('randomness', 0.0)
        rng = effects.get('rng')
        return rng.random() * randomness if (randomness > 0 and rng is not None) else 0.0

    # ── Statistiche (locali al render, non stato dell'oggetto) ──────────────
    @staticmethod
    def new_statistics():
        return {'low_total': 0.0, 'mid_total': 0.0, 'high_total': 0.0, 'total_energy': 0.0,
                'pow_low': 0.0, 'pow_mid': 0.0, 'pow_high': 0.0}

    def update_statistics(self, stats, time_idx):
        t = min(max(int(time_idx), 0), self.norm_bands.shape[1] - 1)
        low, mid, high = self.get_normalized_bands(t)
        stats['low_total'] += low
        stats['mid_total'] += mid
        stats['high_total'] += high
        stats['total_energy'] += low + mid + high
        p_low, p_mid, p_high = self.band_power[:, t]
        stats['pow_low'] += float(p_low)
        stats['pow_mid'] += float(p_mid)
        stats['pow_high'] += float(p_high)

    @staticmethod
    def get_color_percentages(stats):
        """Peso visivo: ogni banda normalizzata sul proprio picco"""
        total = stats['total_energy']
        if total == 0:
            return 0, 0, 0
        return (stats['low_total'] / total * 100,
                stats['mid_total'] / total * 100,
                stats['high_total'] / total * 100)

    @staticmethod
    def get_spectral_shares(stats):
        """Quota reale di energia spettrale per banda (potenza non normalizzata)"""
        total = stats['pow_low'] + stats['pow_mid'] + stats['pow_high']
        if total == 0:
            return 0, 0, 0
        return (stats['pow_low'] / total * 100,
                stats['pow_mid'] / total * 100,
                stats['pow_high'] / total * 100)

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
        """Crea un frame (Figure OO, niente stato globale pyplot)"""
        low_norm, mid_norm, high_norm = self.get_normalized_bands(time_idx)

        if colors is None:
            colors = {
                'low': '#FF0000', 'mid': '#0000FF', 'high': '#FFFFFF', 'bg': '#000000'
            }

        effects = dict(effects) if effects else {'intensity': 1.0, 'speed': 0.1, 'randomness': 0.0}
        effects['rng'] = np.random.default_rng([int(effects.get('seed', 507)), int(time_idx)])

        xlim, ylim = self.get_aspect_ratio_limits(aspect_ratio)

        if resolution_px and dpi:
            figsize = (resolution_px[0] / dpi, resolution_px[1] / dpi)
        else:
            base_width = 10
            figsize = (base_width, base_width * (ylim / xlim))

        fig = Figure(figsize=figsize, facecolor=colors['bg'], dpi=dpi)
        FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])  # a tutto frame: dimensione di output esatta
        ax.set_facecolor(colors['bg'])

        drawers = {
            "waves": self.draw_classic_waves,
            "interference": self.draw_interference_waves,
            "flowing": self.draw_flowing_waves,
            "am": self.draw_am_waves,
            "fm": self.draw_fm_waves,
            "reflected": self.draw_reflected_waves,
            "varied_amplitude": self.draw_varied_amplitude_waves,
            "varied_shape": self.draw_varied_shape_waves,
            "varied_motion": self.draw_varied_motion_waves,
        }
        drawers.get(pattern_type, self.draw_classic_waves)(
            ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim
        )

        if title_settings and title_settings['text']:
            self.draw_title(ax, title_settings, xlim, ylim)

        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.axis('off')

        return fig

    @staticmethod
    def fig_to_rgb(fig, size):
        """Figure -> array RGB uint8 esattamente (h, w, 3); corregge eventuali off-by-one"""
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        w, h = size
        if arr.shape[0] != h or arr.shape[1] != w:
            out = np.empty((h, w, 3), dtype=np.uint8)
            out[:] = arr[0, 0]
            hh, ww = min(h, arr.shape[0]), min(w, arr.shape[1])
            out[:hh, :ww] = arr[:hh, :ww]
            arr = out
        return np.ascontiguousarray(arr)

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
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            random_offset = self._rand(effects)
            wave = y_offset + low * intensity * np.sin(2 * np.pi * (0.3 + i * 0.2) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['low'], linewidth=4*low*intensity, alpha=0.8)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            random_offset = self._rand(effects)
            wave = y_offset + mid * intensity * 0.8 * np.sin(2 * np.pi * (0.8 + i * 0.4) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid*intensity, alpha=0.7)
        
        # Onde acute - rapide e piccole
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            random_offset = self._rand(effects)
            wave = y_offset + high * intensity * 0.6 * np.sin(2 * np.pi * (1.5 + i * 0.6) * x/xlim + time_offset + random_offset)
            ax.plot(x, wave, color=colors['high'], linewidth=(1.5+high)*intensity, alpha=0.9)
    
    def draw_interference_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Pattern di interferenza strutturato - onde che si incrociano come nell'immagine"""
        x = np.linspace(0, xlim, 1000)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Layer 1: Onde ampie (basse frequenze) - rosse/arancioni
        num_low_waves = 4  # numero fisso: niente linee che compaiono/spariscono tra frame
        for i in range(num_low_waves):
            base_freq = 0.4 + i * 0.3
            random_offset = self._rand(effects)
            
            # Onda principale
            y1 = ylim/2 + low * intensity * 2.0 * np.sin(2 * np.pi * base_freq * x/xlim + time_offset + random_offset)
            # Onda interferente con fase diversa
            y2 = ylim/2 + low * intensity * 1.5 * np.sin(2 * np.pi * (base_freq * 1.3) * x/xlim - time_offset * 0.7 + random_offset)
            
            ax.plot(x, y1, color=colors['low'], linewidth=3 + low*2*intensity, alpha=0.7)
            ax.plot(x, y2, color=colors['low'], linewidth=2.5 + low*1.5*intensity, alpha=0.5)
        
        # Layer 2: Onde medie (frequenze medie) - blu/turchesi
        num_mid_waves = 5
        for i in range(num_mid_waves):
            base_freq = 1.0 + i * 0.4
            random_offset = self._rand(effects)
            
            # Pattern di interferenza più complesso
            y1 = ylim/2 + mid * intensity * 1.2 * np.sin(2 * np.pi * base_freq * x/xlim + time_offset * 1.5 + random_offset)
            y2 = ylim/2 + mid * intensity * 0.8 * np.sin(2 * np.pi * (base_freq * 1.6) * x/xlim - time_offset + random_offset)
            y3 = ylim/2 + mid * intensity * 0.6 * np.sin(2 * np.pi * (base_freq * 0.7) * x/xlim + time_offset * 2 + random_offset)
            
            ax.plot(x, y1, color=colors['mid'], linewidth=2 + mid*1.5*intensity, alpha=0.8)
            ax.plot(x, y2, color=colors['mid'], linewidth=1.5 + mid*intensity, alpha=0.6)
            ax.plot(x, y3, color=colors['mid'], linewidth=1 + mid*0.8*intensity, alpha=0.4)
        
        # Layer 3: Onde acute (alte frequenze) - gialle/bianche
        num_high_waves = 8
        for i in range(num_high_waves):
            base_freq = 2.0 + i * 0.5
            random_offset = self._rand(effects)
            
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
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
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
                random_offset = self._rand(effects)
                
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
                random_offset = self._rand(effects)
                
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
                random_offset = self._rand(effects)
                
                wave_y = y_base + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset * 2.5 + random_offset)
                
                alpha_val = 0.6 + 0.4 * ((layer-8)/4)
                line_width = 0.8 + high * intensity * (0.6 + (layer-8)/4)
                
                ax.plot(x, wave_y, color=colors['high'], linewidth=line_width, alpha=alpha_val)
    

    def draw_am_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali con modulazione di ampiezza (AM) per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Low freq - modulazione lenta e ampia
        am_low = (1 + 0.5 * np.sin(2 * np.pi * 0.2 * x/xlim + time_offset)) 
        y_low = ylim*0.3 + low * intensity * am_low * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset + self._rand(effects))
        ax.plot(x, y_low, color=colors['low'], linewidth=3*low*intensity, alpha=0.8)

        # Mid freq - modulazione media
        am_mid = (1 + 0.4 * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset*1.2))
        y_mid = ylim*0.5 + mid * intensity * am_mid * np.sin(2 * np.pi * 0.8 * x/xlim + time_offset*1.5 + self._rand(effects))
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5*mid*intensity, alpha=0.7)

        # High freq - modulazione veloce
        am_high = (1 + 0.3 * np.sin(2 * np.pi * 0.8 * x/xlim + time_offset*2))
        y_high = ylim*0.7 + high * intensity * am_high * np.sin(2 * np.pi * 1.6 * x/xlim + time_offset*2.2 + self._rand(effects))
        ax.plot(x, y_high, color=colors['high'], linewidth=2*high*intensity, alpha=0.9)

    def draw_fm_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali con modulazione di frequenza (FM) per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Low freq - FM lenta
        freq_low = 0.4 + 0.1 * np.sin(2 * np.pi * 0.2 * x/xlim + time_offset)
        y_low = ylim*0.3 + low * intensity * np.sin(2 * np.pi * freq_low * x/xlim + time_offset + self._rand(effects))
        ax.plot(x, y_low, color=colors['low'], linewidth=3*low*intensity, alpha=0.8)

        # Mid freq - FM media
        freq_mid = 0.8 + 0.15 * np.sin(2 * np.pi * 0.3 * x/xlim + time_offset*1.3)
        y_mid = ylim*0.5 + mid * intensity * np.sin(2 * np.pi * freq_mid * x/xlim + time_offset*1.4 + self._rand(effects))
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5*mid*intensity, alpha=0.7)

        # High freq - FM veloce
        freq_high = 1.6 + 0.2 * np.sin(2 * np.pi * 0.5 * x/xlim + time_offset*1.8)
        y_high = ylim*0.7 + high * intensity * np.sin(2 * np.pi * freq_high * x/xlim + time_offset*1.9 + self._rand(effects))
        ax.plot(x, y_high, color=colors['high'], linewidth=2*high*intensity, alpha=0.9)

    def draw_reflected_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali riflesse simmetricamente per le 3 bande"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Funzione helper per specchiare onde (stessa fase casuale per l'onda e il suo riflesso)
        def draw_reflected(y_base, amplitude, freq, color, width, alpha):
            y = amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset + self._rand(effects))
            ax.plot(x, y_base + y, color=color, linewidth=width, alpha=alpha)
            ax.plot(x, y_base - y, color=color, linewidth=width, alpha=alpha)
        
        # Low
        draw_reflected(ylim*0.3, low * intensity, 0.4, colors['low'], 3*low*intensity, 0.8)
        # Mid
        draw_reflected(ylim*0.5, mid * intensity, 0.8, colors['mid'], 2.5*mid*intensity, 0.7)
        # High
        draw_reflected(ylim*0.7, high * intensity, 1.6, colors['high'], 2*high*intensity, 0.9)
    
    def draw_varied_amplitude_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde con ampiezza modulata diversamente per ogni banda"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Low: pulsazione esponenziale
        amp_low = low * intensity * (0.5 + 0.5 * np.exp(np.sin(0.5 * x/xlim + time_offset)))
        y_low = ylim*0.3 + amp_low * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset + self._rand(effects))
        ax.plot(x, y_low, color=colors['low'], linewidth=3, alpha=0.8)

        # Mid: ampiezza a gradini
        steps = np.floor(5 * x/xlim) / 5  # 5 gradini
        amp_mid = mid * intensity * (0.4 + 0.6 * steps)
        y_mid = ylim*0.5 + amp_mid * np.sin(2 * np.pi * 0.8 * x/xlim + time_offset*1.5 + self._rand(effects))
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5, alpha=0.7)

        # High: modulazione lenta
        amp_high = high * intensity * (0.6 + 0.4 * np.sin(0.2 * x/xlim + time_offset*0.5))
        y_high = ylim*0.7 + amp_high * np.sin(2 * np.pi * 1.6 * x/xlim + time_offset*2 + self._rand(effects))
        ax.plot(x, y_high, color=colors['high'], linewidth=2, alpha=0.9)

    def draw_varied_shape_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde con forme diverse per ogni banda"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Low: sinusoide classica
        y_low = ylim*0.3 + low * intensity * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset + self._rand(effects))
        ax.plot(x, y_low, color=colors['low'], linewidth=3, alpha=0.8)

        # Mid: sinusoide + armonica (più appuntita)
        theta = 2 * np.pi * 0.8 * x/xlim + time_offset*1.5 + self._rand(effects)
        y_mid = ylim*0.5 + mid * intensity * (np.sin(theta) + 0.3 * np.sin(2*theta))
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5, alpha=0.7)

        # High: doppio seno (effetto schiacciato)
        y_high = ylim*0.7 + high * intensity * np.sin(np.sin(2 * np.pi * 1.6 * x/xlim + time_offset*2 + self._rand(effects)))
        ax.plot(x, y_high, color=colors['high'], linewidth=2, alpha=0.9)

    def draw_varied_motion_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde con movimenti diversi per ogni banda"""
        x = np.linspace(0, xlim, 800)
        time_offset = self._time_offset(time_idx, effects)
        intensity = effects.get('intensity', 1.0)
        
        # Low: movimento orizzontale standard
        y_low = ylim*0.3 + low * intensity * np.sin(2 * np.pi * 0.4 * x/xlim + time_offset + self._rand(effects))
        ax.plot(x, y_low, color=colors['low'], linewidth=3, alpha=0.8)

        # Mid: movimento diagonale (x e y combinati)
        y_mid = ylim*0.5 + mid * intensity * np.sin(2 * np.pi * 0.8 * (0.7*x/xlim + 0.3*y_low/ylim) + time_offset*1.5 + self._rand(effects))
        ax.plot(x, y_mid, color=colors['mid'], linewidth=2.5, alpha=0.7)

        # High: movimento a zig-zag (inversione di fase)
        phase_mod = np.sign(np.sin(0.5 * time_offset))  # inverte la fase periodicamente
        y_high = ylim*0.7 + high * intensity * np.sin(2 * np.pi * 1.6 * x/xlim + phase_mod * time_offset*2 + self._rand(effects))
        ax.plot(x, y_high, color=colors['high'], linewidth=2, alpha=0.9)
    
    def generate_preview_frames(self, pattern_type, colors, effects, num_frames=12):
        """Genera una griglia di frame preview a bassa risoluzione"""
        preview_times = np.linspace(0, len(self.times) - 1, num_frames, dtype=int)
        frames = []
        for t_idx in preview_times:
            fig = self.create_pattern_frame(
                int(t_idx), pattern_type, colors, effects,
                aspect_ratio="16:9 (Standard)",
                title_settings=None,
                resolution_px=(320, 180),
                dpi=72
            )
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=72, facecolor=fig.get_facecolor())
            buf.seek(0)
            frames.append(buf.read())
        return frames

    def generate_social_report(self, audio_filename, video_title, pattern_type, colors, effects,
                                fps, total_frames, video_quality, aspect_ratio,
                                low_percent, mid_percent, high_percent, spectral_shares):
        """Genera testo stilizzato per social/YouTube da scaricare"""
        pattern_labels = {
            "waves": "Classic Waves",
            "interference": "Structured Interference Waves",
            "flowing": "Horizontal Layered Waves",
            "am": "AM Modulated Waves",
            "fm": "FM Modulated Waves",
            "reflected": "Reflected Waves",
            "varied_amplitude": "Variable Amplitude Waves",
            "varied_shape": "Variable Shape Waves",
            "varied_motion": "Variable Motion Waves"
        }
        pattern_label = pattern_labels.get(pattern_type, pattern_type.title())

        # Dominant frequency band
        spec_low, spec_mid, spec_high = spectral_shares
        dominants = [
            (spec_low,  "low frequencies",  "Red Line",   "LOW"),
            (spec_mid,  "mid frequencies",  "Blue Line",  "MID"),
            (spec_high, "high frequencies", "White Line", "HIGH"),
        ]
        dominant = max(dominants, key=lambda x: x[0])

        intensity = effects.get('intensity', 1.0)
        speed = effects.get('speed', 0.1)
        randomness = effects.get('randomness', 0.0)

        intensity_word   = "delicate"           if intensity  < 0.8 else ("powerful"          if intensity  > 1.5 else "balanced")
        speed_word       = "slow and meditative" if speed     < 0.3 else ("frenetic and intense" if speed  > 1.0 else "dynamic and fluid")
        randomness_word  = "perfectly deterministic" if randomness == 0 else ("with slight organic variations" if randomness < 0.4 else "rich in random variations")

        w, h = self.get_resolution(video_quality, aspect_ratio)

        now = datetime.now()
        date_codec = now.strftime("%Y.%m.%d")
        time_codec = now.strftime("%H:%M")
        duration_mm = int(self.duration // 60)
        duration_ss = int(self.duration % 60)
        duration_codec = f"{duration_mm:02d}:{duration_ss:02d}"

        track_name = audio_filename.rsplit('.', 1)[0] if '.' in audio_filename else audio_filename

        report = f"""#generativeart #audioreactive #frequencyart

Visual Experiments in Code _ 00
{pattern_type.upper()} · {date_codec} · {duration_codec}

An audio visualization generated in real time from the frequencies
of *{track_name}*.

The *{pattern_label}* pattern responds to the song's spectral data:
{dominant[1]} dominate with {dominant[0]:.1f}% of the spectral energy,
creating a {intensity_word}, {speed_word} animation,
{randomness_word}.

:: WAVE EFFECTS

  WAVE TYPE    {pattern_label}
  INTENSITY    {intensity:.1f}x  /  {intensity_word}
  SPEED        {speed:.2f}x  /  {speed_word}
  RANDOMNESS   {randomness*100:.0f}%  /  {randomness_word}

:: COLOR LINE

  Red Line:    LOW   20–250 Hz       {low_percent:.1f}%
  Blue Line:   MID   250–4000 Hz     {mid_percent:.1f}%
  White Line:  HIGH  4000–20000 Hz   {high_percent:.1f}%

  (visual weight: each band is normalised to its own peak)

:: SPECTRAL ENERGY

  LOW   20–250 Hz       {spec_low:.1f}%
  MID   250–4000 Hz     {spec_mid:.1f}%
  HIGH  4000–20000 Hz   {spec_high:.1f}%

  Dominant › {dominant[1]} ({dominant[0]:.1f}%)

:: VIDEO INFO

  RESOLUTION   {w}x{h} px  ·  {aspect_ratio}
  FRAMERATE    {fps} FPS  ·  {total_frames:,} frames
  DURATION     {self.duration:.1f}s
  SAMPLERATE   {self.sr:,} Hz

:: TRACK INFO

  TRACK        *{track_name}*
  DURATION     {duration_codec}
  SAMPLERATE   {self.sr:,} Hz
  RESOLUTION   {w}x{h} px  ·  {aspect_ratio}
  FRAMES       {total_frames:,} @ {fps} FPS
  GENERATED    {date_codec}  {time_codec}

__________________________________________
#audiovisualization #musicvisualizer #audiolinetwo
#waves #soundart #musicart #visualmusic #frequencyart
#audioreactive #waveform #sounddesign #generativeart
__________________________________________

audiovisualization, musicvisualizer, audiolinetwo,
waves, soundart, musicart, visualmusic, frequencyart,
audioreactive, waveform, sounddesign, generativeart,
"""
        return report

    def create_video_no_audio(self, output_path, pattern_type, colors, effects, fps, 
                             aspect_ratio="16:9 (Standard)", video_quality="Media (1280x720)", 
                             title_settings=None, progress_cb=None):
        """Crea un video senza audio, scrivendo i frame in streaming (niente PNG su disco)"""
        stats = self.new_statistics()
        resolution_px = self.get_resolution(video_quality, aspect_ratio)

        total_frames = max(1, int(self.duration * fps))
        time_step = self.duration / total_frames
        last_idx = len(self.times) - 1

        writer = imageio_ffmpeg.write_frames(
            output_path, resolution_px, fps=fps, quality=6,
            macro_block_size=1,            # niente resize a multipli di 16 (1080 -> 1088)
            pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            ffmpeg_log_level="error"
        )
        writer.send(None)  # avvia il generatore
        try:
            for frame_idx in range(total_frames):
                current_time = frame_idx * time_step
                time_idx = int(np.clip(round(current_time * self.sr / self.hop_length), 0, last_idx))

                self.update_statistics(stats, time_idx)
                fig = self.create_pattern_frame(
                    time_idx, pattern_type, colors, effects, aspect_ratio,
                    title_settings, resolution_px=resolution_px, dpi=100
                )
                writer.send(self.fig_to_rgb(fig, resolution_px).tobytes())

                if progress_cb:
                    progress_cb(frame_idx + 1, total_frames)
        finally:
            writer.close()

        return total_frames, resolution_px, stats

    @staticmethod
    def _ffmpeg_exe():
        """ffmpeg incluso in imageio-ffmpeg, altrimenti quello di sistema"""
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return shutil.which('ffmpeg')

    def create_video_with_audio(self, output_path, pattern_type, colors, effects, fps, 
                               audio_filename="Unknown Track", video_quality="Media (1280x720)", 
                               aspect_ratio="16:9 (Standard)", video_title="My Audio Visual", title_settings=None):
        """Crea un video completo con audio e genera report finale"""
        ffmpeg = self._ffmpeg_exe()
        if not ffmpeg:
            st.error("ffmpeg non trovato: aggiungi `imageio-ffmpeg` a requirements.txt "
                     "(oppure `ffmpeg` a packages.txt).")
            return False

        progress_bar = st.progress(0)
        status_text = st.empty()

        def on_progress(done, total):
            progress_bar.progress(done / total)
            status_text.text(f"Generando frame {done}/{total}")

        # Tutti i file intermedi vivono in una directory temporanea che sparisce sempre
        with tempfile.TemporaryDirectory() as work_dir:
            temp_video_path = os.path.join(work_dir, 'video_no_audio.mp4')
            temp_audio_path = os.path.join(work_dir, 'audio.wav')

            try:
                total_frames, resolution_px, stats = self.create_video_no_audio(
                    temp_video_path, pattern_type, colors, effects, fps,
                    aspect_ratio, video_quality, title_settings, progress_cb=on_progress
                )
            except Exception as exc:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Errore durante il rendering dei frame: {exc}")
                return False

            status_text.text("🔊 Unisco audio e video...")

            # Estrai l'audio corrispondente alla durata effettiva
            end_sample = int(self.duration * self.sr)
            audio_segment = self.audio_data[:end_sample]
            if np.max(np.abs(audio_segment)) > 0:
                audio_segment = audio_segment / np.max(np.abs(audio_segment))
            wavfile.write(temp_audio_path, self.sr, audio_segment.astype(np.float32))

            command = [
                ffmpeg, '-y',
                '-i', temp_video_path,
                '-i', temp_audio_path,
                '-c:v', 'copy',              # copia il video senza ri-encodare
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                '-movflags', '+faststart',   # avvio rapido per web/social
                output_path
            ]
            try:
                subprocess.run(command, check=True, capture_output=True)
            except subprocess.CalledProcessError as exc:
                st.error(f"Errore durante la combinazione audio/video: {exc.stderr.decode(errors='replace')}")
                return False
            except OSError as exc:
                st.error(f"Impossibile eseguire ffmpeg: {exc}")
                return False
            finally:
                progress_bar.empty()
                status_text.empty()

        self.show_generation_report(audio_filename, video_title, pattern_type,
                                    colors, effects, fps, total_frames,
                                    video_quality, aspect_ratio, title_settings,
                                    resolution_px, stats)
        return True
    
    def show_generation_report(self, audio_filename, video_title, pattern_type, colors, effects, fps, total_frames, video_quality, aspect_ratio, title_settings, resolution_px, stats):
        """Mostra il report dettagliato della generazione"""
        # Calcola le percentuali dei colori
        low_percent, mid_percent, high_percent = self.get_color_percentages(stats)
        spectral_shares = self.get_spectral_shares(stats)
        spec_low, spec_mid, spec_high = spectral_shares
        
        # Formatta la risoluzione
        final_resolution = f"{resolution_px[0]}x{resolution_px[1]}"
        
        # Mappa nomi pattern
        pattern_names = {
            "waves": "Onde Classiche",
            "interference": "Onde Interferenza Strutturate", 
            "flowing": "Onde Stratificate Orizzontali",
            "am": "Onde AM",
            "fm": "Onde FM",
            "reflected": "Onde Riflesse",
            "varied_amplitude": "Onde Ampiezza Variabile",
            "varied_shape": "Onde Forma Variabile",
            "varied_motion": "Onde Movimento Variabile"
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
            title_info = f"{title_settings['text']} ({title_position}, {title_settings['fontsize']}pt)"
        
        # Crea il report
        report = f"""
## 📊 Audio Visual Report - WAVES EDITION

**🎬 Video Title:** {video_title}  
**🎵 Audio Track:** {audio_filename}  
**⏱️ Duration:** {self.duration:.1f}s  
**🔊 Sample Rate:** {self.sr:,} Hz  
**📺 Resolution:** {final_resolution}  
**📐 Aspect Ratio:** {aspect_ratio}

### 🌈 Color Distribution (visual weight):
- **🔴 Low Frequencies (20-250Hz):** {low_percent:.1f}%
- **🔵 Mid Frequencies (250-4000Hz):** {mid_percent:.1f}%  
- **⚪ High Frequencies (4000-20000Hz):** {high_percent:.1f}%

### 📊 Spectral Energy (real share of the track):
- **Low (20-250Hz):** {spec_low:.1f}%
- **Mid (250-4000Hz):** {spec_mid:.1f}%
- **High (4000-20000Hz):** {spec_high:.1f}%

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
        
        # Salva report social in session_state
        social_report = self.generate_social_report(
            audio_filename, video_title, pattern_type,
            colors, effects, fps, total_frames,
            video_quality, aspect_ratio, low_percent, mid_percent, high_percent,
            spectral_shares
        )
        st.session_state['social_report'] = social_report

        # Anche come info success
        st.success(f"""
        ✅ **Video Wave generato con successo!**
        
        **Distribuzione Colori:**
        🔴 Basse: {low_percent:.1f}% | 🔵 Medie: {mid_percent:.1f}% | ⚪ Acute: {high_percent:.1f}%
        
        **Dettagli:** {total_frames:,} frames • {fps} FPS • {self.duration:.1f}s • Wave: {pattern_names.get(pattern_type, pattern_type)} • {final_resolution}
        **Effetti:** Intensità {intensity_desc} • Velocità {effects.get('speed', 0.1)}x • Casualità {effects.get('randomness', 0.0)*100:.0f}%
        """)

def load_audio(audio_bytes, filename):
    """Carica l'audio da bytes. Se librosa non lo decodifica (es. m4a) ripiega su ffmpeg -> wav."""
    suffix = os.path.splitext(filename or '')[1].lower() or '.wav'
    with tempfile.TemporaryDirectory() as work_dir:
        path = os.path.join(work_dir, 'input' + suffix)
        with open(path, 'wb') as f:
            f.write(audio_bytes)
        try:
            return librosa.load(path, sr=None, mono=True)
        except Exception:
            ffmpeg = AudioVisualizer._ffmpeg_exe()
            if not ffmpeg:
                raise
            wav_path = os.path.join(work_dir, 'decoded.wav')
            subprocess.run([ffmpeg, '-y', '-i', path, '-vn', '-acodec', 'pcm_f32le', wav_path],
                           check=True, capture_output=True)
            return librosa.load(wav_path, sr=None, mono=True)


@st.cache_resource(show_spinner=False, max_entries=2)
def get_visualizer(file_hash, _audio_bytes, filename):
    """Analisi audio una sola volta per file (non a ogni rerun di Streamlit)"""
    audio_data, sr = load_audio(_audio_bytes, filename)
    return AudioVisualizer(audio_data, sr)


def show_image(container, data):
    """st.image: use_container_width è deprecato in favore di width="stretch"."""
    try:
        container.image(data, width="stretch")
    except Exception:
        container.image(data, use_container_width=True)

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo WAVES by Loop507",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Inizializza session_state
    for _key in ('run_preview', 'preview_frames', 'create_video',
                 'video_bytes', 'video_filename', 'video_path', 'social_report'):
        if _key not in st.session_state:
            st.session_state[_key] = None

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
        ["waves", "interference", "flowing", "am", "fm", "reflected", 
         "varied_amplitude", "varied_shape", "varied_motion"],
        help="Scegli il tipo di visualizzazione wave",
        format_func=lambda x: {
            "waves": "🌊 Onde Classiche",
            "am": "📡 Onde AM (Ampiezza Modulata)",
            "fm": "🎛️ Onde FM (Frequenza Modulata)",
            "reflected": "🪞 Onde Riflesse",
            "interference": "🔄 Onde Interferenza", 
            "flowing": "💫 Onde Stratificate",
            "varied_amplitude": "📈 Onde Ampiezza Variabile",
            "varied_shape": "🔷 Onde Forma Variabile",
            "varied_motion": "↕️ Onde Movimento Variabile"
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
    
    # Seed: stesso seed = stessi frame (casualità riproducibile)
    seed_value = st.sidebar.number_input("Seed", min_value=0, max_value=2**31 - 1, value=507, step=1,
                                         help="Rende riproducibile la casualità: stesso seed = stesso video")

    # Preparazione effetti
    effects = {
        'intensity': intensity_multiplier,
        'speed': speed_multiplier,
        'randomness': randomness_factor,
        'seed': int(seed_value)
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
            # Carica file audio (analisi in cache: si rifà solo se cambia il file)
            audio_bytes = uploaded_file.getvalue()
            try:
                visualizer = get_visualizer(hashlib.sha1(audio_bytes).hexdigest(),
                                            audio_bytes, uploaded_file.name)
            except Exception as exc:
                st.error(f"Impossibile leggere il file audio: {exc}")
                st.stop()
            duration = visualizer.duration
            sr = visualizer.sr
            
            # Prepara colori
            colors = {
                'low': color_low,
                'mid': color_mid, 
                'high': color_high,
                'bg': bg_color
            }
            
        st.success(f"✅ Audio caricato! Durata: {duration:.1f}s, Sample Rate: {sr}Hz")

        # ── Bottoni azione ──────────────────────────────────────────────
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔍 Preview Rapida"):
                st.session_state['run_preview'] = True
                st.session_state['preview_frames'] = None  # reset
        with col2:
            if st.button("🎥 Crea Video Wave", help="Genera un video della visualizzazione wave"):
                st.session_state['create_video'] = True
                st.session_state['video_bytes'] = None   # reset
                st.session_state['video_filename'] = None
                st.session_state['social_report'] = None
        with col3:
            pattern_labels_ui = {
                "waves": "🌊 Onde Classiche",
                "interference": "🔄 Interferenza",
                "flowing": "💫 Stratificate",
                "am": "📡 AM",
                "fm": "🎛️ FM",
                "reflected": "🪞 Riflesse",
                "varied_amplitude": "📈 Ampiezza Var.",
                "varied_shape": "🔷 Forma Var.",
                "varied_motion": "↕️ Movimento Var."
            }
            st.write(f"Wave selezionata: **{pattern_labels_ui.get(pattern_type, pattern_type)}**")

        # ── PREVIEW a bassa risoluzione ─────────────────────────────────
        if st.session_state.get('run_preview'):
            with st.spinner("🔍 Generando preview (12 frame a 320×180)..."):
                preview_frames = visualizer.generate_preview_frames(
                    pattern_type, colors, effects, num_frames=12
                )
                st.session_state['preview_frames'] = preview_frames
                st.session_state['run_preview'] = False

        if st.session_state.get('preview_frames'):
            st.markdown("#### 🔍 Preview — campioni dal brano")
            cols = st.columns(4)
            for i, frame_bytes in enumerate(st.session_state['preview_frames']):
                show_image(cols[i % 4], frame_bytes)

        # ── CREAZIONE VIDEO ──────────────────────────────────────────────
        if st.session_state.get('create_video'):
            with st.spinner("🎥 Creazione video wave in corso (potrebbe richiedere alcuni minuti)..."):
                old_video = st.session_state.get('video_path')
                if old_video and os.path.exists(old_video):
                    os.remove(old_video)
                st.session_state['video_path'] = None

                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    video_path = tmpfile.name

                audio_filename_str = uploaded_file.name if uploaded_file.name else "Unknown Track"

                success = visualizer.create_video_with_audio(
                    video_path, pattern_type, colors, effects, frame_rate,
                    audio_filename_str, video_quality, aspect_ratio, video_title, title_settings
                )

                if success:
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()

                    slug = pattern_type.replace(" ", "_")
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_filename = f"audioline_wave_{slug}_{ts}.mp4"

                    st.session_state['video_bytes'] = video_bytes
                    st.session_state['video_filename'] = video_filename
                    st.session_state['video_path'] = video_path
                else:
                    st.error("Errore nella creazione del video wave.")
                    if os.path.exists(video_path):
                        os.remove(video_path)

            st.session_state['create_video'] = False

        # ── DOWNLOAD VIDEO (stabile, fuori dal bottone) ──────────────────
        if st.session_state.get('video_bytes'):
            st.video(st.session_state['video_path'])
            st.download_button(
                label="📥 Scarica Video Wave",
                data=st.session_state['video_bytes'],
                file_name=st.session_state['video_filename'],
                mime="video/mp4",
                key="dl_video"
            )

        # ── DOWNLOAD REPORT SOCIAL (stabile, fuori dal bottone) ──────────
        if st.session_state.get('social_report'):
            st.markdown("---")
            st.markdown("#### 📄 Report Social / YouTube")
            st.text_area(
                "Anteprima report",
                value=st.session_state['social_report'],
                height=250,
                disabled=True
            )
            ts_report = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📋 Scarica Report Social (.txt)",
                data=st.session_state['social_report'].encode('utf-8'),
                file_name=f"audioline_report_{ts_report}.txt",
                mime="text/plain",
                key="dl_report"
            )
    
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
        
        ### 📈 **Onde Ampiezza Variabile**
        Ogni banda ha una modulazione di ampiezza diversa: pulsazione esponenziale per le basse, gradini discreti per le medie, modulazione lenta per le alte.
        
        ### 🔷 **Onde Forma Variabile**
        Forme d'onda diverse per ogni banda: sinusoidi classiche per le basse, onde appuntite per le medie, e onde schiacciate per le alte.
        
        ### ↕️ **Onde Movimento Variabile**
        Movimenti distinti per ogni banda: orizzontale per le basse, diagonale per le medie, e zig-zag per le alte.
        
        ### 📡 **Onde AM / 🎛️ FM**
        Modulazione di ampiezza (AM) o di frequenza (FM) diversa per ogni banda.
        
        ### 🪞 **Onde Riflesse**
        Ogni onda è specchiata simmetricamente attorno alla sua linea base.
        
        **🎨 Caratteristiche:**
        - **9 stili wave unici** basati sulla tua immagine
        - **Colori personalizzabili** per ogni banda di frequenza
        - **Report dettagliato** con distribuzione colori ed energia spettrale reale
        - **Seed** per render riproducibili (stessa casualità, stesso video)
        - **Aspect ratio multipli:** 16:9, 1:1, 9:16
        - **Qualità video HD** fino a 1920x1080
        
        **Come usare:**
        1. Carica un file audio dalla sidebar
        2. Scegli il tipo di wave tra le 9 opzioni
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
        
        demo_fig = Figure(figsize=(14, 12), facecolor='black')
        ax1, ax2, ax3 = demo_fig.subplots(3, 1)
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
        ax3.set_title('💫 Esempio: Onde Fluide', color='white', fontsize=14, pad=20)
        ax3.axis('off')
        
        demo_fig.tight_layout()
        st.pyplot(demo_fig, clear_figure=True)

if __name__ == "__main__":
    main()
