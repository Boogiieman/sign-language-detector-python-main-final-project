import os
from PIL import Image

def mirror_images_in_folder(folder_path):
    # Get list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    num_images = len(image_files)

    # Iterate through each image
    for i, image_file in enumerate(image_files):
        # Open the image
        with Image.open(os.path.join(folder_path, image_file)) as img:
            # Mirror the image
            mirrored_img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

            # Save the mirrored image with a new name
            new_name = str(num_images) + os.path.splitext(image_file)[1]
            mirrored_img.save(os.path.join(folder_path, new_name))
            
            # Increment the count of images
            num_images += 1

def mirror_images_in_all_folders(root_folder):
    # Recursively traverse through all subdirectories
    for dirpath, _, _ in os.walk(root_folder):
        # Mirror images in each directory
        mirror_images_in_folder(dirpath)

if __name__ == "__main__":
    root_folder = r"data"
    mirror_images_in_all_folders(root_folder)
    print("Images mirrored and saved successfully.")
