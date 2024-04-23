import cv2
from plotly.offline import plot
import plotly.graph_objects as go
from tqdm import tqdm
from statistics import median
import numpy as np
from scipy.signal import find_peaks

# open videos/video1.MOV

video = cv2.VideoCapture("videos/video1.MOV")

# Get the duration and the number of frames per second
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count / fps
print(f"FPS: {fps}")
print(f"Frame count: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

# The aim is to detecte when the heart beats, so all frames are red
# because the user put his finger on the camera and the flash is on
# but when the heart beats, the color of the frames change to black (or dark red)
# I want to detect this change of color to detect the heart beats
# I will use the mean of the pixel values of the frames to detect this change
# First, we will just plot the mean of the pixel values of the frames

means = []
for _ in tqdm(range(int(frame_count))):
    ret, frame = video.read()
    if not ret:
        break
    means.append(frame.mean())

# compute the median of the means
median_mean = median(means)
print(f"Median of the means: {median_mean}")
# Get the first index where the mean - median_mean changes sign
index = 0
first_sign = np.sign(means[0] - median_mean)
for i, mean in enumerate(means):
    if np.sign(mean - median_mean) != first_sign:
        index = i
        break

print(f"Index: {index}")

frames = [means[i] for i in range(index, len(means))]

# Plot the mean of the pixel values of the frames
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(frames))), y=frames, mode="lines", name="Mean of the pixel values"))
# display the median
# fig.add_trace(go.Scatter(x=[0, len(means)], y=[median_mean, median_mean], mode="lines", name="Median of the means"))
# fig.add_trace(go.Scatter(x=frames, y=[means[i] for i in frames], mode="markers", name="Frames"))
fig.update_layout(title="Mean of the pixel values of the frames")
plot(fig, filename="mean_of_pixel_values.html")

# detect the peaks
peaks, _ = find_peaks(frames, height=median_mean)

# print peaks on the plot
fig.add_trace(go.Scatter(x=peaks, y=[frames[i] for i in peaks], mode="markers", name="Peaks"))
plot(fig, filename="mean_of_pixel_values.html")

# Get 2 highest peaks, get the number of frames between them and compute the heart rate
peaks = sorted(peaks, key=lambda x: frames[x], reverse=True)
two_highest_peaks = peaks[:2]
frames_between_peaks = two_highest_peaks[1] - two_highest_peaks[0]
heart_rate = 60 * fps / frames_between_peaks
print(f"Heart rate: {heart_rate:.2f} bpm")

# Display the two highest peaks on the plot
fig.add_trace(go.Scatter(x=two_highest_peaks, y=[frames[i] for i in two_highest_peaks], mode="markers", name="Two highest peaks"))
plot(fig, filename="mean_of_pixel_values.html")
