import scipy
import torch
from pipeline_audioldm2 import AudioLDM2Pipeline

repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# This Works ??!!!!?????
# prompt_1 = "The vibrant beat of Brazilian samba drums."

# prompt_2 = "An electric guitar solo"

prompt_1 = "An electric guitar solo"

prompt_2 = "An electric guitar solo"

negative_prompt = "Low quality."

# set the seed for generator
generator = torch.Generator("cuda").manual_seed(0)

# run the generation
audio = pipe(
     prompt_1,
     prompt_2,
     negative_prompt=negative_prompt,
     num_inference_steps=200,
     audio_length_in_s=10.0,
     num_waveforms_per_prompt=1,
     generator=generator,
 ).audios

# save the best audio sample (index 0) as a .wav file
scipy.io.wavfile.write("techno.wav", rate=16000, data=audio[0])