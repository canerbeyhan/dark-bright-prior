import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_dark_channel(image, patch_size=15):
    dark_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(dark_channel, np.ones((patch_size, patch_size), dtype=np.uint8))
    return dark_channel

def find_bright_channel(image, patch_size=15):
    bright_channel = np.max(image, axis=2)
    bright_channel = cv2.dilate(bright_channel, np.ones((patch_size, patch_size), dtype=np.uint8))
    return bright_channel

def estimate_atmospheric_light(dark_channel, percentage=0.1):
    # Flatten the dark channel to a 1D array and sort in descending order.
    sorted_dark_channel = np.sort(dark_channel.ravel())[::-1]
    
    # Calculate the number of pixels to consider based on the percentage.
    num_pixels = int(dark_channel.size * percentage / 100)
    
    # Take the top 'num_pixels' pixels to estimate the atmospheric light.
    atmospheric_light = np.max(sorted_dark_channel[:num_pixels])
    return atmospheric_light

def main():
    # Replace 'your_image_path' with the path to your input image.
    image_path = r'C:\Users\user\Desktop\bright-dark prior\test.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Unable to load the image.")
        return
    
    # Pre-process the image by scaling pixel values to the range [0, 1].
    image = image.astype(np.float32) / 255.0
    
    # Set the patch size for dark and bright channel computation.
    patch_size = 15
    
    # Find the dark and bright channels of the image.
    dark_channel = find_dark_channel(image, patch_size=patch_size)
    bright_channel = find_bright_channel(image, patch_size=patch_size)
    
    # Resize the dark and bright channels to fit within the screen.
    screen_res = 1920, 1080  # Change this to your screen resolution if needed.
    scale_percent = min(screen_res[0] / dark_channel.shape[1], screen_res[1] / dark_channel.shape[0]) * 100
    dim = (int(dark_channel.shape[1] * scale_percent / 100), int(dark_channel.shape[0] * scale_percent / 100))
    resized_dark_channel = cv2.resize(dark_channel, dim, interpolation=cv2.INTER_AREA)
    resized_bright_channel = cv2.resize(bright_channel, dim, interpolation=cv2.INTER_AREA)
    
    # Estimate the atmospheric light from the dark channel.
    atmospheric_light = estimate_atmospheric_light(dark_channel)
    print("Estimated Atmospheric Light:", atmospheric_light)
    
    # Create a resizable figure for displaying the images.
    fig = plt.figure(figsize=(8, 6))  # Adjust the size as needed.

    # Display the resized dark and bright channels as grayscale images.
    plt.subplot(121)
    plt.imshow(resized_dark_channel, cmap='gray')
    plt.title('Dark Channel')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(resized_bright_channel, cmap='gray')
    plt.title('Bright Channel')
    plt.axis('off')

    # Show the figure on the screen.
    plt.show()

if __name__ == '__main__':
    main()