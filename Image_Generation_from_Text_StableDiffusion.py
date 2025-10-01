from diffusers import StableDiffusionPipeline
import os
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.to("cuda")
prompt = "an aerial image of a plantation  infested with pests in two to three areas . "
image = pipe(prompt).images[0]
import matplotlib.pyplot as plt

save_folder = "generated_images"
os.makedirs(save_folder, exist_ok=True)

# Define save path
save_path = os.path.join(save_folder, "plantation_pests.png")

# Save the image
image.save(save_path)

# Optional: Show the image
import matplotlib.pyplot as plt
plt.imshow(image)
plt.axis("off")
plt.show()