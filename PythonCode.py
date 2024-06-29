# Group 15 - Music emotion recognition and recommendation system
# 0 Calm
# 1 Happy
# 2 Angry
# 3 Anticipation
# 4 Sad
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
def extract_features(song_path):
# Load the audio file
y, sr = librosa.load(song_path, duration=30) # Adjust the duration as needed
# Extract features
features = []
# 1. Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=None)
features.append(tempo)
# 2. Spectral centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
features.extend(np.mean(centroid, axis=1))
# 3. Zero-crossing rate
zcr = librosa.feature.zero_crossing_rate(y)
features.append(np.mean(zcr))
# 4. MFCCs (13 coefficients)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
features.extend(np.mean(mfcc, axis=1))
# 5. Chroma feature
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
features.extend(np.mean(chroma, axis=1))
# 6. Spectral contrast
contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
features.extend(np.mean(contrast, axis=1))
# 7. Spectral rolloff
6
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
features.append(np.mean(rolloff))
# 8. Spectral bandwidth
bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
features.extend(np.mean(bandwidth, axis=1))
# 9. Mel-scaled spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
features.extend(np.mean(mel_spec, axis=1))
return features
folder_path = "D:\\Main\\Sri\\Amrita ECE\\SEM 4\\19ECE284 - DSP Lab\\Project\\Data\\Tamil"
# List to store the extracted features for all files
all_features = []
# Iterate over the files in the folder
for subfolder in os.listdir(folder_path):
subpath = os.path.join(folder_path, subfolder)
for filename in os.listdir(subpath):
# Check if the file has a supported audio extension
if filename.endswith('.wav') or filename.endswith('.mp3'): # Add more supported extensions if needed
# Construct the full file path
file_path = os.path.join(subpath, filename)
# Extract features for the current file
features = extract_features(file_path)
# Append the features to the list
all_features.append(features)
features_file = open('SongFeaturesTamil.csv', 'w')
writer = csv.writer(features_file, delimiter=',')
writer.writerows(all_features)
features_file.close()
features_file = open('SongFeatures.csv', 'r')
reader = csv.reader(features_file)
songs_features = []
for i in reader:
if len(i) != 0:
songs_features.append(i)
features_file.close()
# Function to perform clustering
def perform_clustering(feature_vectors, num_clusters):
# Initialize the K-means model
kmeans = KMeans(n_clusters=num_clusters)
# Fit the model to the feature vectors
kmeans.fit(feature_vectors)
# Return the cluster labels
return kmeans
7
num_clusters = 2 # Number of emotion clusters
X_train, X_test = train_test_split(songs_features, test_size=0.2, random_state=42)
# Perform clustering
cluster = perform_clustering(songs_features, num_clusters)
audio_files = []
folder_path = 'D:\\Main\\Sri\\Amrita ECE\SEM 4\\19ECE284 - DSP Lab\Project\Data\genres'
for subfolder in os.listdir(folder_path):
subpath = os.path.join(folder_path, subfolder)
for filename in os.listdir(subpath):
# Check if the file has a supported audio extension
if filename.endswith('.wav') or filename.endswith('.mp3'): # Add more supported extensions if needed
# Construct the full file path
file_path = os.path.join(subpath, filename)
# Extract features for the current file
audio_files.append(file_path)
print(audio_files)
def predict_emotion(new_audio_file):
# Extract features from the new audio file
new_features = extract_features(new_audio_file)
# Perform clustering on the new features using the trained model
new_cluster_label = cluster.predict([new_features])
print(cluster.predict([new_features]))
return new_cluster_label
import tkinter as tk
from tkinter import filedialog
import pygame
def recommend_similar_songs(song_path, cluster_labels, audio_files):
# Get the cluster label for the given song
song_emotion = predict_emotion(song_path)
print(song_emotion)
# Find indices of songs in the same cluster
similar_song_indices = [i for i, label in enumerate(cluster_labels) if label == song_emotion]
# Get the paths of similar songs
similar_songs = [audio_files[i] for i in similar_song_indices]
return similar_songs
def browse_file():
# Open file dialog to select a song
file_path = filedialog.askopenfilename(filetypes=[("Audio Files", ".wav"), ("Audio Files", ".mp3")])
if file_path:
# Call the recommend_similar_songs function to get similar songs
similar_songs = recommend_similar_songs(file_path, cluster.labels_, audio_files)
#print("Similar Songs:")
#for song in similar_songs:
8
# print(song)
#music_player.listbox_songs.delete(0, tk.END)
#
#for song_name in similar_songs:
# music_player.listbox_songs.insert(tk.END, song_name)
# Play the given song
#pygame.mixer.music.load(similar_songs[0])
#pygame.mixer.music.play()
return similar_songs
similar_songs = recommend_similar_songs(file_path, cluster.labels_, audio_files)
def play_music():
pygame.mixer.music.unpause()
def pause_music():
pygame.mixer.music.pause()
# Initialize the pygame mixer
pygame.mixer.init()
root = tk.Tk()
# Create a button to browse for a song
class MusicPlayer():
def _init_(self, master, song_list):
self.master = master
self.master.title("Music Player")
self.master.geometry("800x400")
self.song_list = song_list
self.current_song_index = len(song_list)
self.create_widgets()
def browse_get_similar(self):
similar = browse_file()
self.listbox_songs.delete(0, tk.END)
for song_name in similar:
self.listbox_songs.insert(tk.END, song_name)
def create_widgets(self):
self.heading_label = tk.Label(self.master, text="Music Recommendation System", font=("Arial", 20, "bold"))
self.heading_label.pack()
self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_get_similar)
self.browse_button.pack(pady=20)
self.label_song = tk.Label(self.master, text="Similar Songs:")
self.label_song.pack()
self.listbox_songs = tk.Listbox(self.master, width=100)
self.listbox_songs.pack()
9
#self.button_add = tk.Button(self.master, text="Add Song", command=self.add_song)
#self.button_add.pack()
self.button_play = tk.Button(self.master, text="Play", command=self.play_song)
self.button_play.pack()
preexisting_songs = self.song_list
for song_name in preexisting_songs:
self.listbox_songs.insert(tk.END, song_name)
self.button_pause = tk.Button(self.master, text="Pause", command=self.pause_song)
self.button_pause.pack()
def add_song(self):
song_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
if song_path:
song_name = os.path.basename(song_path)
self.song_list.append((song_name, song_path))
self.listbox_songs.insert(tk.END, song_name)
def play_song(self):
selected_index = self.listbox_songs.curselection()
song_path = self.listbox_songs.get(selected_index[0])
print(song_path)
if self.song_list:
pygame.mixer.init()
pygame.mixer.music.load(song_path)
pygame.mixer.music.play()
def pause_song(self):
pygame.mixer.music.pause()
root.title("Song Recommendation System")
similar_songs_name = []
#print(similar_songs_name)
music_player = MusicPlayer(root, c_songs)
# Start the main event loop
root.mainloop()
# Perform PCA
pca = PCA(n_components=2) # Specify the number of components (in this case, 2)
X_pca = pca.fit_transform(songs_features) # Apply PCA to the dataset
# Access the principal components and explained variance ratio
components = pca.components_ # Principal components
explained_variance_ratio = pca.explained_variance_ratio_ # Explained variance ratio
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster.labels_)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.savefig('emotions.png')
plt.show()
