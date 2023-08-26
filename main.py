# https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5

import soundata
import pandas as pd

dataset = soundata.initialize('urbansound8k')
# dataset.download()                                  # download the dataset
# dataset.validate()                                  # validate that all the expected files are there
# example_clip = dataset.choice_clip()                # choose a random example clip
# print(example_clip)                                 # see the available data

df = pd.DataFrame()
for key, audio_item in dataset.load_clips().items():
    relative_path = audio_item.audio_path
    relative_path = audio_item.audio_path[audio_item.audio_path.rfind('fold')-1:]
    class_id = audio_item.class_id

    new_item = pd.DataFrame([relative_path, class_id])
    df = pd.concat([df, new_item], ignore_index=True)

print(df)