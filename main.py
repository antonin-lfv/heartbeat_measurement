import streamlit as st
import cv2
import plotly.graph_objects as go
from tqdm import tqdm
from statistics import median
import numpy as np
from scipy.signal import find_peaks

st.set_page_config(page_title="Détection du rythme cardiaque", layout="wide")

st.title("Détection du rythme cardiaque à partir d'une vidéo")

col1, col2 = st.columns((3, 1))

video_file = col1.file_uploader("Choisissez une vidéo", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Sauvegarde temporaire de la vidéo
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Ouverture de la vidéo avec OpenCV
    video = cv2.VideoCapture("temp_video.mp4")

    # Obtention de la durée, du nombre d'images par seconde et du nombre total d'images
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps

    col1.markdown("### Détails de la vidéo")
    col1.write(f"- FPS : {fps}")
    col1.write(f"- Nombre d'images : {frame_count}")
    col1.write(f"- Durée : {duration:.2f} secondes")
    col1.divider()

    # Affichage de la preview de la vidéo
    col2.markdown("### Preview de la vidéo")
    col2.video("temp_video.mp4")

    if col1.button("Lancer le calcul du rythme cardiaque"):
        # Calcul de la moyenne des valeurs de pixels pour chaque image
        means = []
        progress_bar = col1.progress(0)

        for i in tqdm(range(int(frame_count))):
            ret, frame = video.read()
            if not ret:
                break
            means.append(frame.mean())
            progress_bar.progress((i + 1) / frame_count)

        # Calcul de la médiane des moyennes
        median_mean = median(means)

        # Détection du changement de signe
        index = 0
        first_sign = np.sign(means[0] - median_mean)
        for i, mean in enumerate(means):
            if np.sign(mean - median_mean) != first_sign:
                index = i
                break

        frames = [means[i] for i in range(index, len(means))]

        # Tracé du signal filtré
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(frames))), y=frames, mode="lines", name="Signal filtré"))

        # Ajout de la moyenne
        fig.add_hline(y=sum(frames) / len(frames),
                      line_dash="dash",
                      line_color="red",
                      annotation_text="Moyenne",
                      annotation_position="bottom right")

        # Ajout de Q3
        Q3 = np.percentile(frames, 75)
        fig.add_hline(y=Q3,
                      line_dash="dash", line_color="orange", annotation_text="Q3",
                      annotation_position="bottom right")

        # Détection des pics au dessus de Q3
        peaks, _ = find_peaks(frames, height=Q3)
        fig.add_trace(go.Scatter(x=peaks, y=[frames[i] for i in peaks],
                                 mode="markers",
                                 name="Pics détectés",
                                 marker=dict(color="green")))

        # Sélection du plus hauts pics pour calculer le rythme cardiaque avec son suivant
        highest_peaks_index = np.argsort([frames[i] for i in peaks])
        next_peak = peaks[highest_peaks_index[-1] + 1]
        two_highest_peaks = [peaks[highest_peaks_index[-1]], next_peak]

        # Afficher les pics sélectionnés
        fig.add_trace(go.Scatter(x=two_highest_peaks,
                                 y=[frames[i] for i in two_highest_peaks],
                                 mode="markers",
                                 name="Pics sélectionnés"))

        frames_between_peaks = two_highest_peaks[1] - two_highest_peaks[0]
        heart_rate = 60 * fps / frames_between_peaks

        st.balloons()
        col1.title(f"Rythme cardiaque : {heart_rate:.2f} bpm")
        col1.plotly_chart(fig, use_container_width=True)
