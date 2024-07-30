# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Define the generator model
class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            # Input is Z, going into a convolution
            torch.nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            # State size. (512) x 4 x 4
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            # State size. (256) x 8 x 8
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            # State size. (128) x 16 x 16
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            # State size. (64) x 32 x 32
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # Output size. (3) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    return image, edges

# Function to generate an image using the generator
def generate_image(generator, z_dim=100):
    # Generate random noise
    z = torch.randn(1, z_dim, 1, 1)
    # Generate image from noise
    with torch.no_grad():
        generated_image = generator(z).detach().cpu()
    return generated_image

# Main function
def main(image_path, model_path):
    # Preprocess the image
    original_image, edges = preprocess_image(image_path)

    # Initialize and load the pre-trained generator
    generator = Generator()
    generator.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    generator.eval()

    # Generate the reconstructed image
    reconstructed_image = generate_image(generator)

    # Save and display the generated image
    save_image(reconstructed_image, 'reconstructed_image.png')

    # Display the original, edge-detected, and reconstructed images
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed Image')
    plt.imshow(np.transpose(reconstructed_image[0].numpy(), (1, 2, 0)))
    plt.show()

# Set paths to the image and model
image_path = 'path_to_your_image.jpg'
model_path = 'path_to_pretrained_model.pth'

# Run the main function
main(image_path, model_path)
