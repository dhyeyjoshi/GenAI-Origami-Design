import torch
from model_loader import preload_models_from_standard_weights
import numpy as np

# Path to the pre-trained model checkpoint
ckpt_path = '../data/v1-5-pruned-emaonly.ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the models
models = preload_models_from_standard_weights(ckpt_path, device)
encoder = models['encoder']
decoder = models['decoder']
diffusion = models['diffusion']
clip = models['clip']

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class OrigamiDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg') or fname.endswith('.JPEG')]
        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = image.resize((512, 512))
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0 * 2 - 1  # Normalize to [-1, 1]
        return image_tensor

image_dir = '../origami_images/'
dataset = OrigamiDataset(image_dir)

if len(dataset) == 0:
    raise ValueError(f"No images found in the directory {image_dir}")

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class StableDiffusionFineTuner(nn.Module):
    def __init__(self, encoder, decoder, diffusion, clip):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion = diffusion
        self.clip = clip
        self.loss_fn = nn.MSELoss()
    
    def forward(self, image, noise, context):
        latents = self.encoder(image, noise)
        output = self.diffusion(latents, context)
        return self.decoder(output)
    
    def compute_loss(self, images, noise, context):
        latents = self.encoder(images, noise)
        noisy_latents = self.diffusion.add_noise(latents, noise)
        outputs = self.decoder(noisy_latents)
        return self.loss_fn(outputs, images)

def generate_clip_context(clip_model, images):
    # Assuming the input for CLIP is tokenized in a specific way
    batch_size, _, _, _ = images.shape
    tokens = torch.randint(0, 49408, (batch_size, 77)).to(device)  # Simulating token generation
    context = clip_model(tokens)
    return context

# Initialize the model and optimizer
model = StableDiffusionFineTuner(encoder, decoder, diffusion, clip).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        images = batch.to(device)
        # Adjust noise dimensions
        noise = torch.randn(images.shape[0], 8, images.shape[2] // 8, images.shape[3] // 8).to(device)
        
        # Generate CLIP context (embedding)
        context = generate_clip_context(clip, images)
        
        # Print the shape of tensors before computing loss
        print(f"Batch images shape: {images.shape}")
        print(f"Noise shape: {noise.shape}")
        print(f"Context shape: {context.shape}")
        
        # Compute loss
        try:
            loss = model.compute_loss(images, noise, context)
        except RuntimeError as e:
            print(f"Error during loss computation: {e}")
            print(f"Image shape: {images.shape}")
            print(f"Noise shape: {noise.shape}")
            print(f"Context shape: {context.shape}")
            raise e
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Save the fine-tuned model
torch.save(model.state_dict(), 'fine_tuned_stable_diffusion.pth')
