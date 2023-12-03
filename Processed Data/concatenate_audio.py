import os
import random
import numpy as np
from pydub import AudioSegment


def create_audio_and_list(
    folder1, folder2, output_audio_folder, output_list_folder, num_audios=500
):
    os.makedirs(output_audio_folder, exist_ok=True)
    os.makedirs(output_list_folder, exist_ok=True)
    for i in range(373, num_audios):
        combined_audio = AudioSegment.empty()
        list_of_numbers = []

        for _ in range(10):
            choice = random.randint(0, 1)
            current_folder = folder1 if choice == 0 else folder2
            chosen_file = random.choice(os.listdir(current_folder))
            audio = AudioSegment.from_file(os.path.join(current_folder, chosen_file))

            combined_audio += audio
            list_of_numbers.append(choice)

        # Ensure the combined audio is 20 seconds long
        combined_audio = combined_audio[:20000]

        # Save the audio
        audio_filename = f"audio_{i}.mp3"
        combined_audio.export(
            os.path.join(output_audio_folder, audio_filename), format="mp3"
        )

        # Save the list
        npy_filename = f"audio_{i}.npy"
        np.save(
            os.path.join(output_list_folder, npy_filename), np.array(list_of_numbers)
        )


# Example usage
create_audio_and_list(
    "Processed Data/Child Speech",
    "Processed Data/Adult Speech",
    "Processed Data/Concatenat 2sec",
    "Processed Data/Concatenat 2sec labels",
)
