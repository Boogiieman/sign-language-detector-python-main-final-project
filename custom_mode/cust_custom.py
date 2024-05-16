import os
import cv2

DATA_DIR = 'custom_mode\cust_data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    
# Find the number of existing folders inside "data"
existing_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
number_of_folders = len(existing_folders)

# Create a new folder with a name as the number of existing folders + 1
new_folder_name = str(number_of_folders)
new_folder_path = os.path.join(DATA_DIR, new_folder_name)
os.makedirs(new_folder_path, exist_ok=True)

number_of_classes = 1  # Since we are creating only one new folder
dataset_size = 200

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width to 1280
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height to 720

# Create a fullscreen window
#cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print('Press "Q" to start capturing for class {}'.format(new_folder_name))

# Wait for 'q' key press to start capturing
while True:
    ret, frame = cap.read()
    cv2.putText(frame, 'Press "Q" to start capturing!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q') or cv2.waitKey(25) == ord('Q'):
        break

print('Collecting data for class {}'.format(new_folder_name))

counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)

    # Save images inside the newly created folder with filenames from 0 to 99
    cv2.imwrite(os.path.join(new_folder_path, '{}.jpg'.format(counter)), frame)

    counter += 1

cap.release()
cv2.destroyAllWindows()
