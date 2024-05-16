#warning
#warning
#warning
#specific custom folder crop, remember to specify the folder at the bottom portion of the code
import os
import cv2

def crop_and_resize_images(folder_path):
    # Get list of files in the folder
    file_list = os.listdir(folder_path)

    # Loop through each file in the folder
    for file_name in file_list:
        # Check if the file is an image (you can add more image extensions if needed)
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            image_path = os.path.join(folder_path, file_name)
            image = cv2.imread(image_path)

            # Get image dimensions
            height, width = image.shape[:2]

            # Calculate new height for resizing
            new_height = 340

            # Calculate crop region
            top = int((height - new_height) / 2)
            bottom = height - top

            # Crop the image
            cropped_image = image[top:bottom, :]

            # Resize the image to 640x340
            resized_image = cv2.resize(cropped_image, (640, new_height))

            # Save the resized image
            cv2.imwrite(image_path, resized_image)

            print(f"{file_name} processed successfully.")

# Specify the folder path containing the images
folder_path = r"specify folder"

# Call the function to crop and resize images
crop_and_resize_images(folder_path)