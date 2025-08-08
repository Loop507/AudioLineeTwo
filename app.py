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

# Classe AudioVisualizer
class AudioVisualizer:
    def __init__(self, audio_data, sr, duration=30):
        self.audio_data = audio_data
        self.sr = sr
        self.duration = min(duration, len(audio_data) / sr)
        self.setup_frequency_analysis()
        
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
        
        # Definisci bande di frequenza
        self.freq_bins = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        self.low_freq_idx = np.where((self.freq_bins >= 20) & (self.freq_bins <= 250))[0]
        self.mid_freq_idx = np.where((self.freq_bins >= 250) & (self.freq_bins <= 4000))[0]
        self.high_freq_idx = np.where((self.freq_bins >= 4000) & (self.freq_bins <= 20000))[0]
        
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
        max_val = max(low, mid, high) if max(low, mid, high) > 0 else 1
        low_norm = (low / max_val) * 0.8 + 0.2
        mid_norm = (mid / max_val) * 0.8 + 0.2
        high_norm = (high / max_val) * 0.8 + 0.2
        return low_norm, mid_norm, high_norm
    
    def create_pattern_frame(self, time_idx, pattern_type="blocks", colors=None, effects=None):
        """Crea un frame del pattern basato sulle frequenze"""
        low_norm, mid_norm, high_norm = self.get_normalized_bands(time_idx)
        
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
        
        fig, ax = plt.subplots(figsize=(14, 10), facecolor=colors['bg'])
        ax.set_facecolor(colors['bg'])
        
        if pattern_type == "blocks":
            self.draw_blocks_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
        elif pattern_type == "lines":
            self.draw_lines_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
        elif pattern_type == "waves":
            self.draw_waves_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
        elif pattern_type == "geometric":
            self.draw_geometric_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
            
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        return fig
    
    def draw_blocks_pattern(self, ax, low, mid, high, colors, effects, time_idx):
        """Pattern a blocchi colorati"""
        # Applica moltiplicatore dimensione
        size_mult = effects['size_mult']
        
        # Blocchi grandi per frequenze basse
        for i in range(int(low * 15)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.5, 2.0) * low * size_mult
            height = np.random.uniform(0.3, 1.5) * low * size_mult
            color = colors['low']
            alpha = (0.3 + low * 0.7) * effects['alpha']
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
        
        # Blocchi medi per frequenze medie
        for i in range(int(mid * 25)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.2, 1.0) * mid * size_mult
            height = np.random.uniform(0.2, 1.0) * mid * size_mult
            color = colors['mid']
            alpha = (0.4 + mid * 0.6) * effects['alpha']
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
        
        # Blocchi piccoli per frequenze alte
        for i in range(int(high * 40)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.1, 0.5) * high * size_mult
            height = np.random.uniform(0.1, 0.5) * high * size_mult
            color = colors['high']
            alpha = (0.5 + high * 0.5) * effects['alpha']
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
    
    def draw_lines_pattern(self, ax, low, mid, high, colors, effects, time_idx):
        """Pattern di linee orizzontali"""
        # Applica moltiplicatore dimensione
        size_mult = effects['size_mult']
        movement = effects['movement']
        
        # Linee spesse per basse
        for i in range(int(low * 8)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 3)
            x_end = x_start + np.random.uniform(2, 7) * low * size_mult
            ax.plot([x_start, x_end], [y, y], 
                   color=colors['low'], linewidth=8*low*size_mult, alpha=0.7*effects['alpha'])
        
        # Linee medie
        for i in range(int(mid * 12)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 4)
            x_end = x_start + np.random.uniform(1, 5) * mid * size_mult
            ax.plot([x_start, x_end], [y, y], 
                   color=colors['mid'], linewidth=4*mid*size_mult, alpha=0.6*effects['alpha'])
        
        # Linee sottili per acute
        for i in range(int(high * 20)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 5)
            x_end = x_start + np.random.uniform(0.5, 3) * high * size_mult
            ax.plot([x_start, x_end], [y, y], 
                   color=colors['high'], linewidth=(1+high)*size_mult, alpha=0.8*effects['alpha'])
    
    def draw_waves_pattern(self, ax, low, mid, high, colors, effects, time_idx):
        """Pattern ondulatorio"""
        x = np.linspace(0, 10, 200)
        size_mult = effects['size_mult']
        movement = effects['movement']
        time_offset = time.time() * movement
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = 2 + i * 2
            wave = y_offset + low * np.sin(2 * np.pi * (0.5 + i * 0.3) * x + time_offset)
            ax.plot(x, wave, color=colors['low'], linewidth=5*low*size_mult, alpha=0.7*effects['alpha'])
        
        # Onde medie
        for i in range(4):
            y_offset = 1.5 + i * 1.5
            wave = y_offset + mid * 0.7 * np.sin(2 * np.pi * (1 + i * 0.5) * x + time_offset)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid*size_mult, alpha=0.6*effects['alpha'])
        
        # Onde acute - rapide e piccole
        for i in range(6):
            y_offset = 1 + i * 1.2
            wave = y_offset + high * 0.5 * np.sin(2 * np.pi * (2 + i * 0.8) * x + time_offset)
            ax.plot(x, wave, color=colors['high'], linewidth=(1+high)*size_mult, alpha=0.8*effects['alpha'])
    
    def draw_geometric_pattern(self, ax, low, mid, high, colors, effects, time_idx):
        """Pattern geometrico misto"""
        size_mult = effects['size_mult']
        movement = effects['movement']
        time_offset = time.time() * movement
        
        # Cerchi grandi per basse
        for i in range(int(low * 8)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            radius = (0.3 + low * 0.8) * size_mult
            circle = plt.Circle((x, y), radius, 
                              color=colors['low'], alpha=(0.3+low*0.4)*effects['alpha'], fill=True)
            ax.add_patch(circle)
        
        # Triangoli per medie
        for i in range(int(mid * 12)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            size = (0.2 + mid * 0.6) * size_mult
            triangle = np.array([[x, y+size], [x-size, y-size], [x+size, y-size]])
            ax.plot(triangle[:, 0], triangle[:, 1], 
                   color=colors['mid'], linewidth=(2+mid*3)*size_mult, alpha=0.6*effects['alpha'])
        
        # Stelle piccole per acute
        for i in range(int(high * 20)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            size = (0.1 + high * 0.3) * size_mult
            ax.scatter(x, y, s=(50+high*100)*size_mult, c=colors['high'], 
                      marker='*', alpha=(0.7+high*0.3)*effects['alpha'])
    
    def create_video_no_audio(self, output_path, pattern_type, colors, effects, duration, fps):
        """Crea un video senza audio"""
        total_frames = int(duration * fps)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Crea una directory temporanea per i frame
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        # Genera tutti i frame
        for frame_idx in range(total_frames):
            # Calcola indice temporale
            time_idx = int((frame_idx / total_frames) * self.magnitude.shape[1])
            
            # Crea frame
            fig = self.create_pattern_frame(time_idx, pattern_type, colors, effects)
            
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
    
    def create_video_with_audio(self, output_path, pattern_type, colors, effects, duration, fps):
        """Crea un video completo con audio"""
        # Crea un video temporaneo senza audio
        temp_video_path = output_path.replace('.mp4', '_no_audio.mp4')
        self.create_video_no_audio(temp_video_path, pattern_type, colors, effects, duration, fps)
        
        # Crea un file audio temporaneo
        temp_audio_path = output_path.replace('.mp4', '.wav')
        
        # Estrai l'audio corrispondente alla durata
        start_sample = 0
        end_sample = int(duration * self.sr)
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
                '-strict', 'experimental',
                output_path
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            st.error(f"Errore durante la combinazione audio/video: {e.stderr.decode()}")
            return False
        finally:
            # Pulisci i file temporanei
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        
        return True

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per theme dinamico
def get_theme_css(is_dark_mode=True):
    if is_dark_mode:
        return """
        <style>
        .main { background-color: #0e1117; }
        .stApp { background-color: #0e1117; }
        .title { color: #00ffff; font-size: 3rem; text-align: center; font-weight: bold; margin-bottom: 2rem; text-shadow: 0 0 20px #00ffff; }
        .subtitle { color: #ff00ff; text-align: center; font-size: 1.2rem; margin-bottom: 2rem; }
        .download-btn { background: linear-gradient(45deg, #00ffff, #ff00ff); color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; }
        </style>
        """
    else:
        return """
        <style>
        .main { background-color: #ffffff; }
        .stApp { background-color: #ffffff; }
        .title { color: #0066cc; font-size: 3rem; text-align: center; font-weight: bold; margin-bottom: 2rem; text-shadow: 0 0 10px #0066cc; }
        .subtitle { color: #cc0066; text-align: center; font-size: 1.2rem; margin-bottom: 2rem; }
        .download-btn { background: linear-gradient(45deg, #0066cc, #cc0066); color: white; border: none; padding: 10px 20px; border-radius: 5px; font-weight: bold; cursor: pointer; }
        </style>
        """

def main():
    st.sidebar.header("🎛️ Controlli")
    
    # Selezione tema
    theme_mode = st.sidebar.selectbox(
        "🎨 Tema Interfaccia",
        ["🌙 Dark Mode", "☀️ Light Mode"],
        help="Scegli tema scuro o chiaro"
    )
    
    is_dark = theme_mode == "🌙 Dark Mode"
    st.markdown(get_theme_css(is_dark), unsafe_allow_html=True)
    
    # Titolo principale
    st.markdown('<h1 class="title">🎵 AudioLineTwo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">BY LOOP507</p>', unsafe_allow_html=True)
    
    # Upload file audio
    uploaded_file = st.sidebar.file_uploader(
        "Carica file audio",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Formati supportati: WAV, MP3, M4A, FLAC"
    )
    
    # Selezione pattern
    pattern_type = st.sidebar.selectbox(
        "Tipo di Pattern",
        ["blocks", "lines", "waves", "geometric"],
        help="Scegli il tipo di visualizzazione"
    )
    
    # Controlli colori personalizzati
    st.sidebar.subheader("🎨 Colori Pattern")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        color_low = st.color_picker("Freq. Basse", "#00BFFF", help="Colore per frequenze basse")
        color_mid = st.color_picker("Freq. Medie", "#00CED1", help="Colore per frequenze medie")
    with col2:
        color_high = st.color_picker("Freq. Acute", "#40E0D0", help="Colore per frequenze acute")
        if is_dark:
            bg_color = st.color_picker("Sfondo", "#16213e", help="Colore di sfondo")
        else:
            bg_color = st.color_picker("Sfondo", "#f0f2f6", help="Colore di sfondo")
    
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
    
    # Durata visualizzazione
    duration = st.sidebar.slider("Durata (secondi)", 5, 60, 20)
    
    # Qualità video
    video_quality = st.sidebar.selectbox("Qualità Video", ["Bassa (720p)", "Media (1080p)", "Alta (2K)"], index=1)
    
    if uploaded_file is not None:
        with st.spinner("🎵 Caricamento e analisi audio..."):
            # Carica file audio
            audio_bytes = uploaded_file.read()
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
            
            # Crea visualizzatore
            visualizer = AudioVisualizer(audio_data, sr, duration)
            
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
            
        st.success(f"✅ Audio caricato! Durata: {len(audio_data)/sr:.1f}s, Sample Rate: {sr}Hz")
        
        # Controlli playback
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🎬 Avvia Visualizzazione"):
                st.session_state.start_viz = True
        
        with col2:
            frame_rate = st.selectbox("FPS", [10, 15, 20, 30], index=2)
        
        with col3:
            st.write(f"Pattern: **{pattern_type.upper()}**")
            
        with col4:
            if st.button("🎥 Crea Video", help="Genera un video della visualizzazione"):
                st.session_state.create_video = True
        
        # Visualizzazione in tempo reale
        if 'start_viz' in st.session_state and st.session_state.start_viz:
            placeholder = st.empty()
            
            total_frames = int(duration * frame_rate)
            progress_bar = st.progress(0)
            
            for frame in range(total_frames):
                # Calcola indice temporale
                time_idx = int((frame / total_frames) * visualizer.magnitude.shape[1])
                
                # Crea frame
                fig = visualizer.create_pattern_frame(time_idx, pattern_type, colors, effects)
                
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
                # Determina la qualità
                quality_map = {
                    "Bassa (720p)": (1280, 720),
                    "Media (1080p)": (1920, 1080),
                    "Alta (2K)": (2560, 1440)
                }
                width, height = quality_map[video_quality]
                
                # Crea file video temporaneo
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    video_path = tmpfile.name
                
                # Crea il video con audio
                success = visualizer.create_video_with_audio(video_path, pattern_type, colors, effects, duration, frame_rate)
                
                if success:
                    # Mostra il video e il pulsante di download
                    st.video(video_path)
                    
                    # Pulsante di download
                    with open(video_path, "rb") as f:
                        video_bytes = f.read()
                    
                    st.download_button(
                        label="📥 Scarica Video con Audio",
                        data=video_bytes,
                        file_name=f"audioline_two_{pattern_type}.mp4",
                        mime="video/mp4"
                    )
                else:
                    st.error("Errore nella creazione del video. Controlla i log per maggiori dettagli.")
                
                # Pulisci lo stato
                st.session_state.create_video = False
    
    else:
        # Schermata iniziale
        st.markdown("""
        ### 🎵 Benvenuto in AudioLineTwo!
        
        Questa applicazione crea pattern visivi dinamici basati sulle frequenze audio:
        
        - **🔊 Frequenze Basse** → Pattern grandi e spessi
        - **🎸 Frequenze Medie** → Pattern di dimensione media  
        - **🎼 Frequenze Acute** → Pattern piccoli e sottili
        
        **Come usare:**
        1. Carica un file audio dalla sidebar
        2. Scegli il tipo di pattern
        3. Avvia la visualizzazione
        
        **Nuova Funzionalità:**
        - **🎥 Crea Video**: Genera e scarica un video della tua visualizzazione con audio
        
        **Pattern disponibili:**
        - **Blocks**: Blocchi rettangolari strutturati
        - **Lines**: Linee orizzontali di spessore variabile
        - **Waves**: Forme ondulatorie dinamiche
        - **Geometric**: Figure geometriche (cerchi, triangoli, stelle)
        
        **Controlli avanzati:**
        - 🎨 **Colori personalizzati** per ogni banda di frequenza
        - 📏 **Dimensioni** regolabili con moltiplicatore
        - 🌊 **Movimento** ondulatorio sincronizzato
        - ✨ **Effetti** glow, sfumature, modalità griglia
        - 🎥 **Qualità video** configurabile (720p, 1080p, 2K)
        """)
        
        # Demo pattern statico
        st.markdown("### 🎨 Anteprima Pattern")
        
        demo_bg = '#16213e' if is_dark else '#f0f2f6'
        demo_fig, demo_ax = plt.subplots(figsize=(14, 8), facecolor=demo_bg)
        demo_ax.set_facecolor(demo_bg)
        
        # Crea demo semplice senza AudioVisualizer
        colors = ['#00BFFF', '#00CED1', '#40E0D0']
        
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
