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
    
    def create_pattern_frame(self, time_idx, pattern_type="waves", colors=None, 
                            aspect_ratio="16:9 (Standard)", title_settings=None, 
                            resolution_px=None, dpi=100):
        """Crea un frame del pattern basato sulle frequenze - SOLO WAVES"""
        low_norm, mid_norm, high_norm = self.get_normalized_bands(time_idx)
        
        # Aggiorna statistiche colori
        self.update_color_statistics(low_norm, mid_norm, high_norm)
        
        # Colori default se non specificati
        if colors is None:
            colors = {
                'low': '#FF0000', 'mid': '#0000FF', 'high': '#FFFFFF', 'bg': '#000000'
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
            self.draw_classic_waves(ax, low_norm, mid_norm, high_norm, colors, time_idx, xlim, ylim)
        elif pattern_type == "interference":
            self.draw_interference_waves(ax, low_norm, mid_norm, high_norm, colors, time_idx, xlim, ylim)
        elif pattern_type == "flowing":
            self.draw_flowing_waves(ax, low_norm, mid_norm, high_norm, colors, time_idx, xlim, ylim)
            
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
    
    def draw_classic_waves(self, ax, low, mid, high, colors, time_idx, xlim, ylim):
        """Pattern ondulatorio classico - originale"""
        x = np.linspace(0, xlim, 500)
        
        # Usa l'indice temporale per sincronizzare le onde con la musica
        time_offset = time_idx * 0.1
        
        # Onde basse - ampie e lente
        for i in range(3):
            y_offset = ylim*0.2 + i * (ylim*0.25)
            wave = y_offset + low * np.sin(2 * np.pi * (0.3 + i * 0.2) * x/xlim + time_offset)
            ax.plot(x, wave, color=colors['low'], linewidth=4*low, alpha=0.8)
        
        # Onde medie
        for i in range(4):
            y_offset = ylim*0.15 + i * (ylim*0.2)
            wave = y_offset + mid * 0.8 * np.sin(2 * np.pi * (0.8 + i * 0.4) * x/xlim + time_offset)
            ax.plot(x, wave, color=colors['mid'], linewidth=3*mid, alpha=0.7)
        
        # Onde acute - rapide e piccole
        for i in range(5):
            y_offset = ylim*0.1 + i * (ylim*0.18)
            wave = y_offset + high * 0.6 * np.sin(2 * np.pi * (1.5 + i * 0.6) * x/xlim + time_offset)
            ax.plot(x, wave, color=colors['high'], linewidth=(1.5+high), alpha=0.9)
    
    def draw_interference_waves(self, ax, low, mid, high, colors, time_idx, xlim, ylim):
        """Pattern di interferenza - onde che si intrecciano (tipo immagine superiore)"""
        x = np.linspace(0, xlim, 800)
        time_offset = time_idx * 0.08
        
        # Onde che si intersecano - stile interferenza
        # Onde rosse (basse frequenze) - grandi ampiezze
        for i in range(4):
            freq1 = 0.5 + i * 0.3
            freq2 = 0.7 + i * 0.2
            y1 = ylim/2 + low * 1.5 * np.sin(2 * np.pi * freq1 * x/xlim + time_offset)
            y2 = ylim/2 + low * 1.2 * np.sin(2 * np.pi * freq2 * x/xlim - time_offset * 1.5)
            
            # Disegna le onde interferenti
            ax.plot(x, y1, color=colors['low'], linewidth=2 + low*2, alpha=0.6)
            ax.plot(x, y2, color=colors['low'], linewidth=2 + low*2, alpha=0.4)
        
        # Onde blu/turchesi (medie frequenze) - frequenze intermedie
        for i in range(6):
            freq1 = 1.0 + i * 0.4
            freq2 = 1.2 + i * 0.3
            y1 = ylim/2 + mid * 1.0 * np.sin(2 * np.pi * freq1 * x/xlim + time_offset * 2)
            y2 = ylim/2 + mid * 0.8 * np.sin(2 * np.pi * freq2 * x/xlim - time_offset)
            
            ax.plot(x, y1, color=colors['mid'], linewidth=1.5 + mid*1.5, alpha=0.7)
            ax.plot(x, y2, color=colors['mid'], linewidth=1.5 + mid*1.5, alpha=0.5)
        
        # Onde gialle/bianche (alte frequenze) - frequenze elevate
        for i in range(8):
            freq1 = 2.0 + i * 0.6
            freq2 = 2.3 + i * 0.5
            y1 = ylim/2 + high * 0.6 * np.sin(2 * np.pi * freq1 * x/xlim + time_offset * 3)
            y2 = ylim/2 + high * 0.4 * np.sin(2 * np.pi * freq2 * x/xlim - time_offset * 2)
            
            ax.plot(x, y1, color=colors['high'], linewidth=1 + high, alpha=0.8)
            ax.plot(x, y2, color=colors['high'], linewidth=1 + high, alpha=0.6)
    
    def draw_flowing_waves(self, ax, low, mid, high, colors, time_idx, xlim, ylim):
        """Pattern di onde fluide - linee sottili e fluide (tipo immagine inferiore)"""
        x = np.linspace(0, xlim, 1000)
        time_offset = time_idx * 0.05
        
        # Onde fluide multiple - stile flusso continuo
        # Layer di base - onde lente e ampie
        for i in range(8):
            phase = i * np.pi / 4
            freq = 0.3 + i * 0.1
            amplitude = low * (0.8 + 0.4 * np.sin(time_offset + phase))
            
            y = ylim/2 + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset + phase)
            
            # Colore sfumato dal rosso al bianco
            alpha_val = 0.3 + 0.4 * (i / 8)
            color_intensity = i / 8
            color = (1.0, color_intensity * 0.5, color_intensity * 0.5)  # Da rosso a rosa-bianco
            
            ax.plot(x, y, color=color, linewidth=0.8 + low*0.5, alpha=alpha_val)
        
        # Layer intermedio - onde medie
        for i in range(12):
            phase = i * np.pi / 6
            freq = 0.6 + i * 0.15
            amplitude = mid * (0.6 + 0.3 * np.sin(time_offset * 1.5 + phase))
            
            y = ylim/2 + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset * 1.5 + phase)
            
            # Gradiente dal blu al ciano
            alpha_val = 0.2 + 0.3 * (i / 12)
            color_intensity = i / 12
            color = (color_intensity * 0.3, 0.5 + color_intensity * 0.5, 1.0)  # Da blu a ciano
            
            ax.plot(x, y, color=color, linewidth=0.6 + mid*0.4, alpha=alpha_val)
        
        # Layer superiore - onde rapide e sottili
        for i in range(15):
            phase = i * np.pi / 8
            freq = 1.2 + i * 0.2
            amplitude = high * (0.4 + 0.2 * np.sin(time_offset * 2.5 + phase))
            
            y = ylim/2 + amplitude * np.sin(2 * np.pi * freq * x/xlim + time_offset * 2.5 + phase)
            
            # Gradiente verso il bianco/giallo
            alpha_val = 0.4 + 0.5 * (i / 15)
            color_intensity = 0.7 + 0.3 * (i / 15)
            color = (color_intensity, color_intensity, color_intensity)  # Verso il bianco
            
            ax.plot(x, y, color=color, linewidth=0.4 + high*0.3, alpha=alpha_val)
    
    def create_video_no_audio(self, output_path, pattern_type, colors, fps, 
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
                time_idx, pattern_type, colors, aspect_ratio, 
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
    
    def create_video_with_audio(self, output_path, pattern_type, colors, fps, 
                               audio_filename="Unknown Track", video_quality="Media (1280x720)", 
                               aspect_ratio="16:9 (Standard)", video_title="My Audio Visual", title_settings=None):
        """Crea un video completo con audio e genera report finale"""
        # Crea un video temporaneo senza audio
        temp_video_path = output_path.replace('.mp4', '_no_audio.mp4')
        total_frames, resolution_px = self.create_video_no_audio(
            temp_video_path, pattern_type, colors, fps, 
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
                                      colors, fps, total_frames, 
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
    
    def show_generation_report(self, audio_filename, video_title, pattern_type, colors, fps, total_frames, video_quality, aspect_ratio, title_settings, resolution_px):
        """Mostra il report dettagliato della generazione"""
        # Calcola le percentuali dei colori
        low_percent, mid_percent, high_percent = self.get_color_percentages()
        
        # Formatta la risoluzione
        final_resolution = f"{resolution_px[0]}x{resolution_px[1]}"
        
        # Mappa nomi pattern
        pattern_names = {
            "waves": "Onde Classiche",
            "interference": "Onde Interferenza", 
            "flowing": "Onde Fluide"
        }
        
        # Prepara info titolo
        title_info = "❌ Disabilitato"
        if title_settings and title_settings['text']:
            title_position = f"{title_settings['v_position']} {title_settings['h_position']}"
            title_info = f"{title_settings['text']} ({title_position}, {title_settings['fontsize']}px)"
        
        # Crea il report
        report = f"""
## 📊 Audio Visual Report - WAVES ONLY

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
        ["waves", "interference", "flowing"],
        help="Scegli il tipo di visualizzazione wave",
        format_func=lambda x: {
            "waves": "🌊 Onde Classiche",
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
                fig = visualizer.create_pattern_frame(time_idx, pattern_type, colors, aspect_ratio, title_settings)
                
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
                    video_path, pattern_type, colors, frame_rate, 
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
                        "interference": "interference_waves", 
                        "flowing": "flowing_waves"
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
        
        ### 🔄 **Onde Interferenza** 
        Onde che si intrecciano e interferiscono tra loro, creando pattern complessi ispirati all'immagine superiore che hai fornito. Ideale per musica elettronica e dinamica.
        
        ### 💫 **Onde Fluide**
        Flusso continuo di onde sottili e fluide che si sovrappongono, ispirato all'immagine inferiore. Perfetto per ambient e musica rilassante.
        
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
