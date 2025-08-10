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
        self.audio = audio_data
        self.sr = sr
        self.duration = duration if duration else librosa.get_duration(y=audio_data, sr=sr)
        self.n_fft = 2048
        self.hop_length = 512
        self.spec = None
        self.freqs = None
        self.times = None
        
        # Calcola lo spettrogramma
        self.compute_spectrogram()
    
    def compute_spectrogram(self):
        S = np.abs(librosa.stft(self.audio, n_fft=self.n_fft, hop_length=self.hop_length))
        self.spec = librosa.amplitude_to_db(S, ref=np.max)
        self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.times = librosa.times_like(self.spec, sr=self.sr, hop_length=self.hop_length)
    
    def get_band_energy(self, time_idx):
        frame_idx = np.argmin(np.abs(self.times - time_idx))
        frame = self.spec[:, frame_idx]
        
        # Divide le frequenze in 3 bande
        low_band = (self.freqs <= 300)
        mid_band = (self.freqs > 300) & (self.freqs <= 3000)
        high_band = (self.freqs > 3000)
        
        low_energy = np.mean(frame[low_band]) if np.any(low_band) else -80
        mid_energy = np.mean(frame[mid_band]) if np.any(mid_band) else -80
        high_energy = np.mean(frame[high_band]) if np.any(high_band) else -80
        
        # Normalizza le energie tra 0 e 1
        low_norm = max(0, (low_energy + 80) / 80)
        mid_norm = max(0, (mid_energy + 80) / 80)
        high_norm = max(0, (high_energy + 80) / 80)
        
        return low_norm, mid_norm, high_norm

    def create_pattern_frame(self, time_idx, pattern_type="waves", colors=None, effects=None,
                            aspect_ratio="16:9 (Standard)", title_settings=None, 
                            resolution_px=None, dpi=100):
        """Crea un frame del pattern basato sulle frequenze - SOLO WAVES con effetti"""
        # Impostazioni predefinite
        if colors is None:
            colors = {
                'low': '#FF0000',    # Rosso
                'mid': '#0000FF',    # Blu
                'high': '#FFFFFF'    # Bianco
            }
            
        if effects is None:
            effects = {
                'speed': 0.1,
                'intensity': 1.0,
                'background': 'black'
            }
            
        if title_settings is None:
            title_settings = {
                'text': "AudioLineTwo WAVES",
                'color': '#FFFFFF',
                'size': 40,
                'position': (0.5, 0.95)
            }
            
        # Calcola le energie delle bande
        low_norm, mid_norm, high_norm = self.get_band_energy(time_idx)
        
        # Configura dimensioni immagine basate su aspect ratio
        aspect_ratios = {
            "16:9 (Standard)": (16, 9),
            "1:1 (Quadrato)": (1, 1),
            "4:3 (TV)": (4, 3),
            "21:9 (Cinema)": (21, 9),
            "9:16 (Verticale)": (9, 16),
            "Personalizzato": (16, 9)  # Default per personalizzato
        }
        
        # Gestione risoluzione
        if resolution_px is None:
            width, height = aspect_ratios[aspect_ratio]
            base_size = 10
            figsize = (base_size * width/height, base_size)
        else:
            width, height = resolution_px
            figsize = (width/dpi, height/dpi)
        
        # Crea figura e axes
        fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=effects.get('background', 'black'))
        ax = fig.add_subplot(111)
        ax.set_facecolor(effects.get('background', 'black'))
        
        # Rimuovi bordi e assi
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Imposta limiti per l'area di disegno
        xlim = figsize[0] * 2
        ylim = figsize[1] * 2
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        
        # Aggiungi titolo se specificato
        if title_settings.get('show', True):
            ax.text(title_settings['position'][0] * xlim, 
                    title_settings['position'][1] * ylim,
                    title_settings['text'],
                    fontsize=title_settings['size'],
                    color=title_settings['color'],
                    ha='center', va='center',
                    alpha=title_settings.get('alpha', 1.0))
        
        # Disegna il pattern wave specifico
        if pattern_type == "waves":
            self.draw_classic_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "interference":
            self.draw_interference_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "flowing":
            self.draw_flowing_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "sine":
            self.draw_sine_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "square":
            self.draw_square_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
        elif pattern_type == "sawtooth":
            self.draw_sawtooth_waves(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx, xlim, ylim)
            
        # Converti in immagine
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf

    # METODI ESISTENTI PER GLI EFFETTI BASE (placeholder)
    def draw_classic_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Disegna onde classiche (placeholder)"""
        pass
        
    def draw_interference_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Disegna onde di interferenza (placeholder)"""
        pass
        
    def draw_flowing_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Disegna onde fluide (placeholder)"""
        pass

    # NUOVI METODI PER GLI EFFETTI WAVE AGGIUNTI
    def draw_sine_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde sinusoidali pulite - versione minimalista"""
        x = np.linspace(0, xlim, 800)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Frequenze base per ogni banda
        low_freq = 0.3
        mid_freq = 0.8
        high_freq = 1.5
        
        # Onde basse - 3 onde principali
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            freq = low_freq * (i + 1) * 0.7
            wave = y_offset + low * intensity * np.sin(2 * np.pi * freq * x/xlim + time_offset)
            ax.plot(x, wave, color=colors['low'], linewidth=4*low, alpha=0.8)
        
        # Onde medie - 4 onde
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            freq = mid_freq * (i + 1) * 0.9
            wave = y_offset + mid * intensity * np.sin(2 * np.pi * freq * x/xlim + time_offset*1.3)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid, alpha=0.7)
        
        # Onde acute - 5 onde
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            freq = high_freq * (i + 1) * 1.2
            wave = y_offset + high * intensity * np.sin(2 * np.pi * freq * x/xlim + time_offset*1.7)
            ax.plot(x, wave, color=colors['high'], linewidth=2*high, alpha=0.9)

    def draw_square_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde quadre con transizioni nette"""
        x = np.linspace(0, xlim, 1000)
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Funzione per generare onde quadre
        def square_wave(t, duty_cycle=0.5):
            """Genera un'onda quadra con duty cycle personalizzato"""
            t_mod = t % (2 * np.pi)
            return np.where(t_mod < 2 * np.pi * duty_cycle, 1, -1)
        
        # Onde basse - ampie e con duty cycle variabile
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            freq = 0.2 + i * 0.15
            t = 2 * np.pi * freq * x/xlim + time_offset
            wave = y_offset + low * intensity * square_wave(t, duty_cycle=0.3 + i*0.1)
            ax.plot(x, wave, color=colors['low'], linewidth=5*low, alpha=0.8)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            freq = 0.5 + i * 0.25
            t = 2 * np.pi * freq * x/xlim + time_offset*1.5
            wave = y_offset + mid * intensity * square_wave(t, duty_cycle=0.4)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid, alpha=0.7)
        
        # Onde acute - rapide e strette
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            freq = 1.2 + i * 0.4
            t = 2 * np.pi * freq * x/xlim + time_offset*2.0
            wave = y_offset + high * intensity * square_wave(t, duty_cycle=0.2 + i*0.05)
            ax.plot(x, wave, color=colors['high'], linewidth=1.5*high, alpha=0.9)

    def draw_sawtooth_waves(self, ax, low, mid, high, colors, effects, time_idx, xlim, ylim):
        """Onde a dente di sega con salti bruschi"""
        x = np.linspace(0, xlim, 1200)
        # Correzione: rimossa la parentesi finale extra qui sotto
        time_offset = time_idx * effects.get('speed', 0.1)
        intensity = effects.get('intensity', 1.0)
        
        # Funzione per generare onde a dente di sega
        def sawtooth_wave(t):
            """Genera un'onda a dente di sega"""
            return 2 * (t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5))
        
        # Onde basse - lente e ampie
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            freq = 0.15 + i * 0.1
            t = 2 * np.pi * freq * x/xlim + time_offset
            wave = y_offset + low * intensity * sawtooth_wave(t)
            ax.plot(x, wave, color=colors['low'], linewidth=5*low, alpha=0.8)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            freq = 0.4 + i * 0.2
            t = 2 * np.pi * freq * x/xlim + time_offset*1.7
            wave = y_offset + mid * intensity * sawtooth_wave(t)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid, alpha=0.7)
        
        # Onde acute - rapide e ripide
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            freq = 0.9 + i * 0.35
            t = 2 * np.pi * freq * x/xlim + time_offset*2.5
            wave = y_offset + high * intensity * sawtooth_wave(t)
            ax.plot(x, wave, color=colors['high'], linewidth=1.5*high, alpha=0.9)

    def show_generation_report(self, audio_filename, video_title, pattern_type, colors, effects, fps, total_frames, video_quality, aspect_ratio, title_settings, resolution_px):
        """Mostra il report dettagliato della generazione"""
        # Mappa nomi pattern (AGGIUNTA NUOVI PATTERN)
        pattern_names = {
            "waves": "Onde Classiche",
            "interference": "Onde Interferenza Strutturate", 
            "flowing": "Onde Stratificate Orizzontali",
            "sine": "Sinusoidali Pure",
            "square": "Onde Quadre",
            "sawtooth": "Onde a Dente di Sega"
        }
        
        # ... [codice esistente invariato] ...
        pass

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo WAVES by Loop507",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.sidebar.header("🌊 Wave Controls")
    
    # Selezione pattern WAVE (AGGIUNTA NUOVI PATTERN)
    pattern_type = st.sidebar.selectbox(
        "Tipo di Onda",
        ["waves", "interference", "flowing", "sine", "square", "sawtooth"],
        help="Scegli il tipo di visualizzazione wave",
        format_func=lambda x: {
            "waves": "🌊 Onde Classiche",
            "interference": "🔄 Onde Interferenza", 
            "flowing": "💫 Onde Fluide",
            "sine": "📈 Sinusoidali Pure",
            "square": "⬛ Onde Quadre",
            "sawtooth": "🔺 Onde a Dente di Sega"
        }[x]
    )
    
    # Controlli colore
    st.sidebar.subheader("🎨 Personalizzazione Colori")
    col_low = st.sidebar.color_picker("Basse Frequenze", "#FF0000")
    col_mid = st.sidebar.color_picker("Medie Frequenze", "#0000FF")
    col_high = st.sidebar.color_picker("Alte Frequenze", "#FFFFFF")
    
    # Effetti speciali
    st.sidebar.subheader("✨ Effetti Speciali")
    speed = st.sidebar.slider("Velocità Animazione", 0.05, 0.5, 0.1, 0.01)
    intensity = st.sidebar.slider("Intensità Onde", 0.5, 2.0, 1.0, 0.1)
    
    # Caricamento file audio
    st.title("🌊 AudioLineTwo - WAVES EDITION")
    uploaded_file = st.file_uploader("Carica un file audio", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Processa l'audio
        audio_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            audio_path = tmp_file.name
            
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Visualizzazione audio
        col1, col2 = st.columns([3, 1])
        with col1:
            st.audio(audio_bytes, format='audio/wav')
            st.write(f"**Durata:** {duration:.2f} secondi")
        
        # Controlli playback (AGGIUNTA NOMI PATTERN)
        with col2:
            pattern_names = {
                "waves": "Onde Classiche",
                "interference": "Onde Interferenza", 
                "flowing": "Onde Fluide",
                "sine": "Sinusoidali Pure",
                "square": "Onde Quadre",
                "sawtooth": "Onde a Dente di Sega"
            }
            st.write(f"Wave: **{pattern_names.get(pattern_type, pattern_type)}**")
            st.write(f"Colore Basse: `{col_low}`")
            st.write(f"Colore Medie: `{col_mid}`")
            st.write(f"Colore Alte: `{col_high}`")
        
        # Visualizza un frame di esempio
        st.subheader("Anteprima Pattern")
        time_idx = st.slider("Seleziona istante temporale", 0.0, duration, duration/2, 0.1)
        
        # Crea visualizzatore
        vis = AudioVisualizer(y, sr)
        colors = {'low': col_low, 'mid': col_mid, 'high': col_high}
        effects = {'speed': speed, 'intensity': intensity}
        
        # Genera e mostra frame
        frame = vis.create_pattern_frame(
            time_idx=time_idx,
            pattern_type=pattern_type,
            colors=colors,
            effects=effects
        )
        st.image(frame, use_column_width=True)
        
    else:
        # Schermata iniziale (AGGIUNTA DESCRIZIONI NUOVI PATTERN)
        st.markdown("""
        ### 🌊 Benvenuto in AudioLineTwo - WAVES EDITION!
        **Un visualizzatore audio avanzato che trasforma il tuo suono in onde ipnotiche.**
        
        ### 📈 **Sinusoidali Pure**
        Onde sinusoidali pulite e minimaliste, perfette per una visualizzazione chiara delle frequenze audio.
        Ideale per musica classica, ambient e suoni armonici.
        
        ### ⬛ **Onde Quadre**
        Pattern con transizioni nette e bruschi cambi di direzione. 
        Eccellente per suoni percussivi, ritmici e musica elettronica con bassi marcati.
        
        ### 🔺 **Onde a Dente di Sega**
        Transizioni ripide e graduali che creano un effetto visivo dinamico. 
        Ottimo per suoni synth, arpeggi e lead elettronici.
        
        **🌊 Altri Tipi di Wave Disponibili:**
        ### 🌊 **Onde Classiche** Forme d'onda tradizionali con curve morbide e naturali. Adatte a qualsiasi genere musicale.
        
        ### 🔄 **Onde Interferenza Strutturate** Pattern complessi creati dall'interazione tra onde multiple. 
        Genera effetti visivi ipnotici e geometrici.
        
        ### 💫 **Onde Stratificate Orizzontali** Onde fluide che si muovono orizzontalmente attraverso lo schermo. 
        Crea un effetto di movimento continuo e rilassante.
        """)
        
        # Demo wave pattern statico (AGGIUNTA NUOVI PATTERN)
        st.markdown("### 🌊 Anteprima Nuovi Stili Wave")
        
        demo_fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), facecolor='black')
        x = np.linspace(0, 16, 500)
        
        # Demo onde sinusoidali
        ax1.set_facecolor('black')
        for i in range(3):
            freq = 0.5 + i * 0.3
            y = 2 + 0.8 * np.sin(2 * np.pi * freq * x/16)
            ax1.plot(x, y, color=['#FF0000', '#0000FF', '#FFFFFF'][i], linewidth=3, alpha=0.8)
        ax1.set_xlim(0, 16)
        ax1.set_ylim(0, 4)
        ax1.set_title('📈 Sinusoidali Pure', color='white', fontsize=14, pad=20)
        ax1.axis('off')
        
        # Demo onde quadre
        ax2.set_facecolor('black')
        for i in range(3):
            freq = 0.4 + i * 0.2
            t = 2 * np.pi * freq * x/16
            y = 2 + 0.8 * np.sign(np.sin(t))
            ax2.plot(x, y, color=['#FF0000', '#0000FF', '#FFFFFF'][i], linewidth=3, alpha=0.8)
        ax2.set_xlim(0, 16)
        ax2.set_ylim(0, 4)
        ax2.set_title('⬛ Onde Quadre', color='white', fontsize=14, pad=20)
        ax2.axis('off')
        
        # Demo onde a dente di sega (CORRETTA)
        ax3.set_facecolor('black')
        for i in range(3):
            freq = 0.6 + i * 0.3
            t = 2 * np.pi * freq * x/16
            # Funzione corretta per onda a dente di sega
            # Correzione: rimossa la parentesi finale extra qui sotto
            y = 2 + 0.8 * (2 * (t/(2*np.pi) - np.floor(t/(2*np.pi) + 0.5)))
            ax3.plot(x, y, color=['#FF0000', '#0000FF', '#FFFFFF'][i], linewidth=3, alpha=0.8)
        ax3.set_xlim(0, 16)
        ax3.set_ylim(0, 4)
        ax3.set_title('🔺 Onde a Dente di Sega', color='white', fontsize=14, pad=20)
        ax3.axis('off')
        
        plt.tight_layout()
        st.pyplot(demo_fig, clear_figure=True)
        plt.close(demo_fig)

if __name__ == "__main__":
    main()
