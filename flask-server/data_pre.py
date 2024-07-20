import os

image_dir = './origami_images'
images = []

for filename in os.listdir(image_dir):
    print(f"Found file: {filename}")
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Corrected to include .jpeg
        image_path = f"./origami_images/{filename}"
        images.append(image_path)
        print(f"Added image path: {image_path}")

print(f"Collected images: {images}")
print(f"total images: {len(images)}")


