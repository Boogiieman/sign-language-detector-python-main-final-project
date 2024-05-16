import os
import cv2

def crop_and_resize_images(folder_path):
    # Loop through all items (files and folders) in the given folder
    for root, dirs, files in os.walk(folder_path):
        # Loop through each file in the current folder
        for file_name in files:
            # Check if the file is an image (you can add more image extensions if needed)
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                # Read the image
                image_path = os.path.join(root, file_name)
                image = cv2.imread(image_path)

                # Get image dimensions
                height, width = image.shape[:2]

                # Calculate new width and height
                new_width = 320
                new_height = 240

                # Calculate crop region
                left = int((width - new_width) / 2)
                right = left + new_width
                top = int((height - new_height) / 2)
                bottom = top + new_height

                # Crop the image
                cropped_image = image[top:bottom, left:right]

                # Save the resized image (overwrite the original)
                cv2.imwrite(image_path, cropped_image)

                print(f"{file_name} processed successfully.")

# Specify the folder path containing the images
folder_path = r"data"

# Call the function to crop and resize images
crop_and_resize_images(folder_path)
