import torch
from diffusers import StableDiffusionPipeline
import os
import time

# Path to your .ckpt file
ckpt_path = "C:/Users/leong/Downloads/AUmodel1.ckpt"

# Set the output directory
output_dir = "C:/Users/leong/Desktop/generated_images"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")

def generate_image_pair(pipe, prompt, negative_prompt, seed, steps, guidance_scale,guidance_scale2):
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    start_time = time.time()
    image1 = pipe(
        prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    end_time = time.time()
    
    print(f"Image 1 generation time: {end_time - start_time:.2f} seconds")

    # Reset the generator to the same seed
    generator = torch.Generator(device=pipe.device).manual_seed(seed)###############################

    start_time = time.time()
    image2 = pipe(
        prompt,
        negative_prompt=negative_prompt if negative_prompt else None,
        num_inference_steps=steps,
        guidance_scale=guidance_scale2,
        generator=generator,
    ).images[0]
    end_time = time.time()
    
    print(f"Image 2 generation time: {end_time - start_time:.2f} seconds")
    
    return image1, image2

try:
    # Load the pipeline directly from the checkpoint
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Move the pipeline to GPU if available
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Enable memory efficient attention if using CUDA
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    print("Model loaded successfully. Ready to generate images!")
    print(f"Images will be saved in: {output_dir}")

    # Initialize a counter for the session
    session_counter = 1

    while True:
        # Get user input for positive prompt
        prompt = input("Enter your positive prompt (or 'quit' to exit): ")
        
        if prompt.lower() == 'quit':
            break

        # Get user input for negative prompt
        negative_prompt = input("Enter your negative prompt (optional, press Enter to skip): ")

        # Get number of image pairs to generate
        while True:
            try:
                num_pairs = int(input("How many image pairs do you want to generate? "))
                if num_pairs > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")

        # Generate image pairs
        for i in range(num_pairs):
            # Generate a random seed
            seed = torch.randint(0, 2**32, (1,)).item()
            steps = 50
            guidance_scale = 7.5
            guidance_scale2= 8

            print(f"\nGenerating pair {i+1}/{num_pairs}")
            print(f"Seed: {seed}, Steps: {steps}, Guidance Scale: {guidance_scale}")

            # Generate the image pair
            image1, image2 = generate_image_pair(pipe, prompt, negative_prompt, seed, steps, guidance_scale,guidance_scale2)

            # Save the images
            filename1 = f"generated_image_{session_counter:04d}_a.png"
            filename2 = f"generated_image_{session_counter:04d}_b.png"
            full_path1 = os.path.join(output_dir, filename1)
            full_path2 = os.path.join(output_dir, filename2)

            image1.save(full_path1)
            image2.save(full_path2)

            print(f"Image pair saved as {filename1} and {filename2}")

            # Increment the counter
            session_counter += 1

        print(f"{num_pairs} image pair(s) generated successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")

print("Thank you for using the Stable Diffusion image generator!")