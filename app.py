import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import librosa
import io
import time
from scipy.signal import find_peaks

# Configurazione pagina
st.set_page_config(
    page_title="AudioLineTwo by Loop507",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato per dark theme
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stApp {
    background-color: #0e1117;
}
.title {
    color: #00ffff;
    font-size: 3rem;
    text-align: center;
    font-weight: bold;
    margin-bottom: 2rem;
    text-shadow: 0 0 20px #00ffff;
}
.subtitle {
    color: #ff00ff;
    text-align: center;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Titolo principale
st.markdown('<h1 class="title">🎵 AudioLineTwo</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">BY LOOP507</p>', unsafe_allow_html=True)

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
    
    def create_pattern_frame(self, time_idx, pattern_type="blocks", colors=None, effects=None):
        """Crea un frame del pattern basato sulle frequenze"""
        low, mid, high = self.get_frequency_bands(time_idx)
        
        # Normalizza i valori
        max_val = max(low, mid, high) if max(low, mid, high) > 0 else 1
        low_norm = (low / max_val) * 0.8 + 0.2
        mid_norm = (mid / max_val) * 0.8 + 0.2
        high_norm = (high / max_val) * 0.8 + 0.2
        
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
            self.draw_structured_blocks(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
        elif pattern_type == "lines":
            self.draw_lines_pattern(ax, low_norm, mid_norm, high_norm, colors, effects, time_idx)
            
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        return fig
    
    def draw_blocks_pattern(self, ax, low, mid, high):
        """Pattern a blocchi colorati (simile all'immagine)"""
        colors_low = ['#00ffff', '#0080ff', '#004080']
        colors_mid = ['#ff00ff', '#ff0080', '#800040']
        colors_high = ['#00ff00', '#80ff00', '#40ff80']
        
        # Blocchi grandi per frequenze basse
        for i in range(int(low * 15)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.5, 2.0) * low
            height = np.random.uniform(0.3, 1.5) * low
            color = np.random.choice(colors_low)
            alpha = 0.3 + low * 0.7
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
        
        # Blocchi medi per frequenze medie
        for i in range(int(mid * 25)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.2, 1.0) * mid
            height = np.random.uniform(0.2, 1.0) * mid
            color = np.random.choice(colors_mid)
            alpha = 0.4 + mid * 0.6
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
        
        # Blocchi piccoli per frequenze alte
        for i in range(int(high * 40)):
            x = np.random.uniform(0, 10)
            y = np.random.uniform(0, 8)
            width = np.random.uniform(0.1, 0.5) * high
            height = np.random.uniform(0.1, 0.5) * high
            color = np.random.choice(colors_high)
            alpha = 0.5 + high * 0.5
            
            rect = Rectangle((x, y), width, height, 
                           facecolor=color, alpha=alpha, edgecolor='none')
            ax.add_patch(rect)
    
    def draw_lines_pattern(self, ax, low, mid, high):
        """Pattern di linee orizzontali"""
        # Linee spesse per basse
        for i in range(int(low * 8)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 3)
            x_end = x_start + np.random.uniform(2, 7) * low
            ax.plot([x_start, x_end], [y, y], 
                   color='cyan', linewidth=8*low, alpha=0.7)
        
        # Linee medie
        for i in range(int(mid * 12)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 4)
            x_end = x_start + np.random.uniform(1, 5) * mid
            ax.plot([x_start, x_end], [y, y], 
                   color='magenta', linewidth=4*mid, alpha=0.6)
        
        # Linee sottili per acute
        for i in range(int(high * 20)):
            y = np.random.uniform(1, 7)
            x_start = np.random.uniform(0, 5)
            x_end = x_start + np.random.uniform(0.5, 3) * high
            ax.plot([x_start, x_end], [y, y], 
                   color='lime', linewidth=1+high, alpha=0.8)
    
    def draw_waves_pattern(self, ax, low, mid, high):
        """Pattern ondulatorio"""
        x = np.linspace(0, 10, 200)
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = 2 + i * 2
            wave = y_offset + low * np.sin(2 * np.pi * (0.5 + i * 0.3) * x + time.time())
            ax.plot(x, wave, color='cyan', linewidth=5*low, alpha=0.7)
        
        # Onde medie
        for i in range(4):
            y_offset = 1.5 + i * 1.5
            wave = y_offset + mid * 0.7 * np.sin(2 * np.pi * (1 + i * 0.5) * x + time.time())
            ax.plot(x, wave, color='magenta', linewidth=3*mid, alpha=0.6)
        
        # Onde acute - rapide e piccole
        for i in range(6):
            y_offset = 1 + i * 1.2
            wave = y_offset + high * 0.5 * np.sin(2 * np.pi * (2 + i * 0.8) * x + time.time())
            ax.plot(x, wave, color='lime', linewidth=1+high, alpha=0.8)
    
    def draw_geometric_pattern(self, ax, low, mid, high):
        """Pattern geometrico misto"""
        # Cerchi grandi per basse
        for i in range(int(low * 8)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            radius = 0.3 + low * 0.8
            circle = plt.Circle((x, y), radius, 
                              color='cyan', alpha=0.3+low*0.4, fill=True)
            ax.add_patch(circle)
        
        # Triangoli per medie
        for i in range(int(mid * 12)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            size = 0.2 + mid * 0.6
            triangle = np.array([[x, y+size], [x-size, y-size], [x+size, y-size]])
            ax.plot(triangle[:, 0], triangle[:, 1], 
                   color='magenta', linewidth=2+mid*3, alpha=0.6)
        
        # Stelle piccole per acute
        for i in range(int(high * 20)):
            x = np.random.uniform(1, 9)
            y = np.random.uniform(1, 7)
            size = 0.1 + high * 0.3
            ax.scatter(x, y, s=50+high*100, c='lime', 
                      marker='*', alpha=0.7+high*0.3)

def main():
    st.sidebar.header("🎛️ Controlli")
    
    # Upload file audio
    uploaded_file = st.sidebar.file_uploader(
        "Carica file audio",
        type=['wav', 'mp3', 'm4a', 'flac'],
        help="Formati supportati: WAV, MP3, M4A, FLAC"
    )
    
    # Selezione pattern
    pattern_type = st.sidebar.selectbox(
        "Tipo di Pattern",
        ["blocks", "lines"],
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
        bg_color = st.color_picker("Sfondo", "#16213e", help="Colore di sfondo")
    
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
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🎬 Avvia Visualizzazione"):
                st.session_state.start_viz = True
        
        with col2:
            frame_rate = st.selectbox("FPS", [10, 15, 20, 30], index=2)
        
        with col3:
            st.write(f"Pattern: **{pattern_type.upper()}**")
        
        # Visualizzazione
        if hasattr(st.session_state, 'start_viz') and st.session_state.start_viz:
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
        
        **Pattern disponibili:**
        - **Blocks**: Blocchi rettangolari strutturati (stile immagine)
        - **Lines**: Linee orizzontali di spessore variabile
        
        **Controlli avanzati:**
        - 🎨 **Colori personalizzati** per ogni banda di frequenza
        - 📏 **Dimensioni** regolabili con moltiplicatore
        - 🌊 **Movimento** ondulatorio sincronizzato
        - ✨ **Effetti** glow, sfumature, modalità griglia
        """)
        
        # Demo pattern statico
        st.markdown("### 🎨 Anteprima Pattern")
        
        demo_fig, demo_ax = plt.subplots(figsize=(14, 8), facecolor='#16213e')
        demo_ax.set_facecolor('#16213e')
        
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
