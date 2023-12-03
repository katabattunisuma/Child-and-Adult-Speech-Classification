from pydub import AudioSegment
import os


def is_silent(audio_path, energy_threshold=-40, duration_threshold=1):
    # Load the audio file using pydub
    audio = AudioSegment.from_file(audio_path)

    # Calculate the energy of the audio
    energy = audio.dBFS

    # Check if the energy level is below the threshold for the given duration
    if energy < energy_threshold and len(audio) > duration_threshold * 1000:
        return True
    else:
        return False


def delete_silent_files(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    silent_files = []

    # Iterate through all audio files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            input_file_path = os.path.join(folder_path, filename)

            # Check if the audio file is silent
            if is_silent(input_file_path):
                silent_files.append(filename)
                os.remove(input_file_path)

    if silent_files:
        print("Silent files identified and deleted:")
        for silent_file in silent_files:
            print(f"- {silent_file}")
    else:
        print("No silent files found.")


# Example usage
folder_path = "/Users/sumakatabattuni/Documents/Child_Speech_Classification/Processed Data/Overlap Speech"
delete_silent_files(folder_path)
