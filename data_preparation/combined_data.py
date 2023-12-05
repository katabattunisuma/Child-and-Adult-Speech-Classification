import os
from pydub import AudioSegment
import random
import numpy as np

# Directories
wav_dir = "Data/Child Speech"
mp3_dir = "Data/Adult Speech"
audio_output_dir = "Data/Combined Speech"
label_output_dir = "Data/Combined Speech Labels"

# Parameters
total_audios = 10000
output_audio_length = 20 * 1000  # 20 seconds in milliseconds
segment_sizes = [
    1500,
    2000,
    2500,
    3000,
]  # Segment sizes in milliseconds (1.5 to 3 seconds)

# Ensure output directories exist
if not os.path.exists(audio_output_dir):
    os.makedirs(audio_output_dir)
if not os.path.exists(label_output_dir):
    os.makedirs(label_output_dir)

# List all WAV and MP3 files
wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
mp3_files = [f for f in os.listdir(mp3_dir) if f.endswith(".mp3")]


# Function to get a random segment
def get_random_segment(audio, segment_size):
    if len(audio) < segment_size:
        # Repeat the audio if it's shorter than the segment size
        repeated_audio = audio * (segment_size // len(audio) + 1)
        return repeated_audio[:segment_size]  # Ensure the segment is the correct size
    else:
        num_segments = len(audio) // segment_size
        start_index = random.randint(0, num_segments - 1) * segment_size
        return audio[start_index : start_index + segment_size]


# Function to create a concatenated audio file with labels
def create_concatenated_audio(wav_file, mp3_file, output_audio_file, output_label_file):
    audio1 = AudioSegment.from_wav(os.path.join(wav_dir, wav_file))
    audio2 = AudioSegment.from_mp3(os.path.join(mp3_dir, mp3_file))
    concatenated_audio = AudioSegment.empty()
    labels = []

    while len(concatenated_audio) < output_audio_length:
        segment_size = random.choice(segment_sizes)
        num_labels = segment_size // 500  # Number of half-second labels per segment

        if random.choice([0, 1]) == 0:
            segment = get_random_segment(audio1, segment_size)
            labels.extend(["0"] * num_labels)
        else:
            segment = get_random_segment(audio2, segment_size)
            labels.extend(["1"] * num_labels)

        concatenated_audio += segment

    # Trim to the exact output length
    concatenated_audio = concatenated_audio[:output_audio_length]

    # Export the audio
    concatenated_audio.export(output_audio_file, format="wav")

    # Save labels as a space-separated string in a txt file
    with open(output_label_file, "w") as label_file:
        label_file.write(" ".join(labels))


# Create concatenated audios and labels
for i in range(total_audios):
    wav_file = wav_files[i % len(wav_files)]
    mp3_file = random.choice(mp3_files)
    base_filename = f"audio_{i+1}"
    output_audio_file = os.path.join(audio_output_dir, base_filename + ".wav")
    output_label_file = os.path.join(label_output_dir, base_filename + ".txt")
    create_concatenated_audio(wav_file, mp3_file, output_audio_file, output_label_file)

print("Audio concatenation and labeling process completed.")
