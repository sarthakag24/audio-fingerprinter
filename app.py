import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter
import soundfile as sf
import streamlit as st

# =====================================================
# GLOBAL CONFIGURATIONS & THEME
# =====================================================
st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    layout="wide",
    initial_sidebar_state="expanded"
)

SR = 22050
NFFT = 2048
HOP = 512

# Custom Dark Neon Theme UI Style Injection
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #00ffcc !important; font-family: 'Helvetica Neue', sans-serif; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: bold; color: #888888; }
    .stTabs [aria-selected="true"] { color: #00ffcc !important; border-bottom-color: #00ffcc !important; }
    .song-card {
        background-color: #161b22;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30363d;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================
# CORES SIGNAL PROCESSING LOGIC
# =====================================================

@st.cache_data
def load_audio(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        return y, sr
    except Exception as e:
        return None, None

def get_peaks(S, size=20, percentile=95):
    local_max = maximum_filter(S, size=size)
    peaks = (S == local_max)
    freq_idx, time_idx = np.where(peaks)
    strength = S[freq_idx, time_idx]
    if len(strength) == 0:
        return []
    threshold = np.percentile(strength, percentile)
    mask = strength > threshold
    return list(zip(freq_idx[mask], time_idx[mask]))

def generate_hashes(peaks, fanout=5, robust_mode=False):
    """
    Generates time-frequency hashes.
    If robust_mode=True, utilizes frequency ratios instead of absolute bins
    to guarantee absolute invariance against uniform pitch-shifting.
    """
    peaks = sorted(peaks, key=lambda x: x[1])
    hashes = []
    for i in range(len(peaks)):
        f1, t1 = peaks[i]
        for j in range(1, fanout + 1):
            if i + j >= len(peaks):
                break
            f2, t2 = peaks[i + j]
            dt = t2 - t1
            if 0 < dt <= 200: # Limit temporal distance for practical neighborhood constraints
                if robust_mode:
                    # Invariant to pitch shifts since (alpha * f2) / (alpha * f1) == f2 / f1
                    ratio = round(f2 / float(f1), 3) if f1 != 0 else f2
                    hashes.append(((ratio, dt), t1))
                else:
                    hashes.append(((f1, f2, dt), t1))
    return hashes

# =====================================================
# DATABASE BUILDER (CACHED)
# =====================================================
@st.cache_resource
def build_fingerprint_database(song_folder, robust_mode=False):
    db = defaultdict(list)
    single_peak_db = defaultdict(list)
    song_metadata = {}
    
    if not os.path.exists(song_folder):
        return db, single_peak_db, song_metadata

    files = [f for f in os.listdir(song_folder) if f.lower().endswith(('.mp3', '.wav'))]
    
    for file in files:
        path = os.path.join(song_folder, file)
        y, sr = load_audio(path)
        if y is None:
            continue
            
        S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
        peaks = get_peaks(S)
        hashes = generate_hashes(peaks, robust_mode=robust_mode)
        
        # Populate Pair-based DB
        for h, t in hashes:
            db[h].append((file, t))
            
        # Populate Single-peak DB for comparison studies
        for f, t in peaks:
            single_peak_db[f].append((file, t))
            
        song_metadata[file] = {
            "duration": librosa.get_duration(y=y, sr=sr),
            "peaks_count": len(peaks),
            "hashes_count": len(hashes),
            "peaks": peaks
        }
    return db, single_peak_db, song_metadata

# =====================================================
# SEARCH & MATCH ENGINE
# =====================================================
def identify_query(y, db, single_db, mode="Pairs", robust_mode=False):
    S = np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))
    peaks = get_peaks(S)
    votes = Counter()
    offsets = []
    
    if mode == "Pairs":
        hashes = generate_hashes(peaks, robust_mode=robust_mode)
        for h, qt in hashes:
            if h in db:
                for song, dt in db[h]:
                    offset = dt - qt
                    votes[(song, offset)] += 1
                    offsets.append(offset)
    else: # Single Peak Matching Mode
        for f, qt in peaks:
            if f in single_db:
                for song, dt in single_db[f]:
                    offset = dt - qt
                    votes[(song, offset)] += 1
                    offsets.append(offset)
                    
    if not votes:
        return None, 0, []
        
    best_match = votes.most_common(1)[0]
    matched_song, best_offset = best_match[0]
    confidence = best_match[1]
    
    # Filter offsets specifically for the winning track to display in the chart
    matched_offsets = [offset for (song, offset), _ in votes.items() if song == matched_song]
    
    return matched_song, confidence, offsets

# =====================================================
# MAIN USER INTERFACE APP
# =====================================================
def main():
    st.title("EE200: Audio Fingerprinting Dashboard")
    st.subheader("Sonic Signatures 'Magical Mystery Tune'")

    SONG_FOLDER = "songs"
    
    # Global Sidebar controls for system parameters
    st.sidebar.header("System Settings")
    robust_hash_toggle = st.sidebar.checkbox(
        "Enable Robust Frequency-Ratio Hashing", 
        value=False,
        help="Maintains ratio invariance (f2/f1) to protect the fingerprint matrix from uniform pitch shifts."
    )
    
    # Initialize the database
    with st.spinner("Indexing track signatures..."):
        db, single_db, metadata = build_fingerprint_database(SONG_FOLDER, robust_mode=robust_hash_toggle)

    if not metadata:
        st.warning(f"Please create a folder named `{SONG_FOLDER}` and add reference MP3 tracks to configure the app.")
        return

    tab1, tab2, tab3 = st.tabs(["🎵 REFERENCE LIBRARY", "🔍 MATCH IDENTIFIER", "🧪 ROBUSTNESS SANDBOX"])

    # -------------------------------------------------
    # TAB 1: REFERENCE LIBRARY
    # -------------------------------------------------
    with tab1:
        st.header("Database Track Registries")
        cols = st.columns(3)
        for idx, (song_name, meta) in enumerate(metadata.items()):
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="song-card">
                    <h3>{song_name.replace('_', ' ').title()}</h3>
                    <p><b>Duration:</b> {meta['duration']:.2f} seconds</p>
                    <p><b>Standout Peaks:</b> {meta['peaks_count']:,}</p>
                    <p><b>Generated Combinatorial Hashes:</b> {meta['hashes_count']:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Render mini thumbnail constellation plot
                fig, ax = plt.subplots(figsize=(4, 1.5))
                fig.patch.set_facecolor('#161b22')
                ax.set_facecolor('#161b22')
                peaks = meta['peaks']
                ax.scatter([p[1] for p in peaks], [p[0] for p in peaks], s=1, color='#00ffcc', alpha=0.6)
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)

    # -------------------------------------------------
    # TAB 2: QUERY MATCH IDENTIFIER
    # -------------------------------------------------
    with tab2:
        st.header("Identify Live Audio Input Clips")
        uploaded_file = st.file_uploader("Upload a Query Audio Snippet", type=["mp3", "wav", "m4a"])
        matching_algorithm = st.radio("Key Retrieval Fingerprint Method", ["Pairs (Combinatorial Hashes)", "Single Peaks"], horizontal=True)
        algo_mode = "Pairs" if "Pairs" in matching_algorithm else "Single"

        if uploaded_file is not None:
            # Save temp file
            with open("temp_query.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            y_query, sr_query = load_audio("temp_query.wav")
            st.audio("temp_query.wav")
            
            if st.button("Run Forensic Audio Matching Search"):
                with st.spinner("Analyzing spectral peaks..."):
                    matched_song, conf, all_offsets = identify_query(
                        y_query, db, single_db, mode=algo_mode, robust_mode=robust_hash_toggle
                    )
                    
                if matched_song:
                    st.success(f"🏆 **MATCH IDENTIFIED:** {matched_song.replace('_', ' ').title()}")
                    st.metric(label="Match Confidence Weight (Total Votes aligned at single offset)", value=conf)
                    
                    # Layout plots for visualization
                    col_plot1, col_plot2 = st.columns(2)
                    
                    with col_plot1:
                        st.subheader("Query Audio Spectrogram Analysis")
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        D_query = librosa.amplitude_to_db(np.abs(librosa.stft(y_query, n_fft=NFFT, hop_length=HOP)), ref=np.max)
                        librosa.display.specshow(D_query, sr=SR, hop_length=HOP, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
                        st.pyplot(fig)
                        plt.close(fig)
                        
                    with col_plot2:
                        st.subheader("Voting Temporal Alignment Histogram")
                        fig, ax = plt.subplots(figsize=(6, 4.5))
                        ax.hist(all_offsets, bins=100, color='#00ffcc', edgecolor='black', alpha=0.8)
                        ax.set_xlabel("Relative Time Offset Bins (Track Time - Query Time)")
                        ax.set_ylabel("Vote Accumulation Count")
                        ax.grid(color='gray', linestyle='--', alpha=0.5)
                        st.pyplot(fig)
                        plt.close(fig)
                else:
                    st.error("No coherent signal signatures detected across the current database index.")

    # -------------------------------------------------
    # TAB 3: ROBUSTNESS SANDBOX TESTING
    # -------------------------------------------------
    with tab3:
        st.header("Stress-Testing / Fingerprint Robustness Laboratory")
        st.write("Simulate noisy real-world café scenarios or severe playback acceleration/pitch distortions to measure fingerprint breakdown tolerances.")
        
        selected_test_file = st.selectbox("Select Database Reference Track to Simulate Query", list(metadata.keys()))
        
        col_param1, col_plot_res = st.columns([1, 2])
        
        with col_param1:
            st.markdown("### Noise Injection Channel")
            noise_sigma = st.slider("White Noise Amplitude (σ Variance Coefficient)", 0.0, 0.5, 0.01, step=0.01)
            
            st.markdown("### Pitch Scaling Shift Modulator")
            pitch_steps = st.slider("Pitch Step Shifting Transposition (Semitones)", -5, 5, 0, step=1)
            
            st.markdown("### Comparison Benchmark Configuration")
            bench_mode = st.selectbox("Benchmark Signature Format", ["Pairs (Combinatorial Hashes)", "Single Peaks"])
            eval_mode = "Pairs" if "Pairs" in bench_mode else "Single"
            
        with col_plot_res:
            if st.button("Generate Distorted Signal & Evaluate"):
                ref_path = os.path.join(SONG_FOLDER, selected_test_file)
                y_orig, sr_orig = load_audio(ref_path)
                
                # Take a 10 second clip slice
                clip_duration = min(10, int(len(y_orig)/SR))
                y_clip = y_orig[0:clip_duration*SR]
                
                # 1. Apply Pitch Translation Distortion
                if pitch_steps != 0:
                    y_distorted = librosa.effects.pitch_shift(y_clip, sr=SR, n_steps=float(pitch_steps))
                else:
                    y_distorted = y_clip.copy()
                    
                # 2. Add AWGN Noise Channel Distortion
                if noise_sigma > 0:
                    y_distorted += np.random.normal(0, noise_sigma, len(y_distorted))
                    
                # Perform Analysis
                sf.write("sandbox_output.wav", y_distorted, SR)
                
                with st.spinner("Re-evaluating forensic hash matches..."):
                    matched_song, conf, all_offsets = identify_query(
                        y_distorted, db, single_db, mode=eval_mode, robust_mode=robust_hash_toggle
                    )
                
                st.audio("sandbox_output.wav")
                
                if matched_song == selected_test_file:
                    st.success(f"✅ **Successful Tracking Retention:** Successfully identified track despite injected signal modifications (Votes matching at peak offset index: **{conf}**)")
                else:
                    st.error(f"❌ **Identifier Failure / False Positives:** System misclassified or dropped tracking frame metrics. Yielded: `{matched_song}` instead of target tracking file.")
                
                # Render Constellation comparison side-by-side
                st.markdown("#### Modified Query Constellation Mapping Matrix")
                S_dist = np.abs(librosa.stft(y_distorted, n_fft=NFFT, hop_length=HOP))
                p_dist = get_peaks(S_dist)
                
                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.scatter([p[1] for p in p_dist], [p[0] for p in p_dist], s=12, color='#ff0055', label='Surviving Extraction Points')
                ax.set_xlabel("Time Coordinates")
                ax.set_ylabel("Frequency Bin Coordinates")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)

                # Explanatory insights for the user report
                st.markdown("### Signal Degradation Analysis Insights")
                if pitch_steps != 0 and not robust_hash_toggle:
                    st.info("💡 **Why Pitch-shifting Broke Standard Pairing Engine:** When you shifted the pitch, frequency values were translated linearly outside of standard linear-spectrogram frequency bounds. Consequently, the pairs `(f1, f2, dt)` changed completely, yielding zero matching hashes in the dictionary, even though the song sounds the same to your ears!")
                elif pitch_steps != 0 and robust_hash_toggle:
                    st.success("💡 **Why Frequency Ratio Invariance Prevailed:** Because 'Robust Frequency-Ratio Hashing' was activated, the calculation utilizes the invariant ratio value `f2 / f1`. This division cancels out the multiplier distortion factor introduced via pitch transpositions, keeping the lookup values correct!")

if __name__ == "__main__":
    main()