import os
from pydub import AudioSegment
import random

# Directories
wav_dir = "Data/Child Speech"
mp3_dir = "Data/Adult Speech"
wav_output_dir = "Data/Overlapped Speech/Child Speech"
mp3_output_dir = "Data/Overlapped Speech/Adult Speech"
overlapped_output_dir = "Data/Overlapped Speech/Overlap "

# Ensure output directories exist
if not os.path.exists(wav_output_dir):
    os.makedirs(wav_output_dir)
if not os.path.exists(mp3_output_dir):
    os.makedirs(mp3_output_dir)
if not os.path.exists(overlapped_output_dir):
    os.makedirs(overlapped_output_dir)

# List all WAV and MP3 files
wav_files = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]
mp3_files = [f for f in os.listdir(mp3_dir) if f.endswith(".mp3")]

# Limit the total number of audio records to 10,000
total_audios = 10000


# Function to overlap two audio segments
def overlap_audios(
    wav_file, mp3_file, output_wav_file, output_mp3_file, output_overlapped_file
):
    audio_wav = AudioSegment.from_wav(os.path.join(wav_dir, wav_file))
    audio_mp3 = AudioSegment.from_mp3(os.path.join(mp3_dir, mp3_file))

    # Find the minimum length of the two audio files
    min_length = min(len(audio_wav), len(audio_mp3))

    # Trim both audio files to this minimum length
    audio_wav = audio_wav[:min_length]
    audio_mp3 = audio_mp3[:min_length]

    # Overlap the audio files by playing them simultaneously
    overlapped_audio = audio_wav.overlay(audio_mp3)

    # Export the original and overlapped audio files
    audio_wav.export(output_wav_file, format="wav")
    audio_mp3.export(output_mp3_file, format="mp3")
    overlapped_audio.export(output_overlapped_file, format="wav")


# Overlap the audios and save them
for i in range(total_audios):
    wav_file = random.choice(wav_files)  # Randomly choose a WAV file
    mp3_file = random.choice(mp3_files)  # Randomly choose an MP3 file
    base_filename = f"audio_{i+1}"
    output_wav_file = os.path.join(wav_output_dir, base_filename + ".wav")
    output_mp3_file = os.path.join(mp3_output_dir, base_filename + ".mp3")
    output_overlapped_file = os.path.join(overlapped_output_dir, base_filename + ".wav")

    overlap_audios(
        wav_file, mp3_file, output_wav_file, output_mp3_file, output_overlapped_file
    )

print("Audio overlapping process completed.")
