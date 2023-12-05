import os
from pydub import AudioSegment


def split_audio(input_path, output_folder, segment_length=2000):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through all audio files in the input folder
    for filename in os.listdir(input_path)[:450]:
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            input_file_path = os.path.join(input_path, filename)
            output_file_prefix = os.path.splitext(filename)[0]

            # Load the audio file using pydub
            audio = AudioSegment.from_file(input_file_path)

            # Calculate the number of segments
            num_segments = int(len(audio) / segment_length)

            # Split the audio into segments
            for i in range(num_segments):
                start_time = i * segment_length
                end_time = (i + 1) * segment_length
                segment = audio[start_time:end_time]

                # Save the segment
                output_segment_path = os.path.join(
                    output_folder, f"{output_file_prefix}_{start_time}-{end_time}.wav"
                )
                segment.export(output_segment_path, format="wav")


# Example usage
input_folder = "Data/Overlapped Speech/Overlap "
output_folder = "Processed Data/Overlap Speech"
split_audio(input_folder, output_folder, segment_length=2000)
