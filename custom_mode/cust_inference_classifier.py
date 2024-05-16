# LOCKED MODE ON RIGHT CLICK AND LEFT CLICK

import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('custom_mode\cust_model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    
    0: ' HI',           #1
    1: ' GOOD MORNING',          #2
    2: ' MY',          #3
    3: ' FRIEND',          #4
    4: ' .',          #5
    5: ' WHATS UP?' ,          #6
    6: '  ',         #Space  gesture 7
    7: 'Backspace',  # Backspace gesture hi-five 
    8: 'Reset',      #Reset gesture (horn)
    
}

word = ""
collect_gesture = True  # Set initial mode to gesture mode

font_path = 'Manjari-Regular.ttf'
font_size = 0.5
font = ImageFont.truetype(font_path, int(font_size * 100))

engine = pyttsx3.init()

# Button properties
button_rect = (10, 50, 100, 40)  # (x, y, width, height)

def on_button_click(event, x, y, flags, param):
    global word, collect_gesture
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_rect[0] < x < button_rect[0] + button_rect[2] and button_rect[1] < y < button_rect[1] + button_rect[3]:
            engine.say(word)
            engine.runAndWait()
    elif event == cv2.EVENT_RBUTTONDOWN:  # Right click to toggle mode
        collect_gesture = not collect_gesture
        if collect_gesture:
            print("Locked mode")
        else:
            print("Unlock mode")

# Create the window and maximize it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Set the callback function for mouse events
cv2.setMouseCallback('frame', on_button_click)

# Main loop
last_backspace_time = time.time()
last_save_time = time.time()  # Initialize last save time 1 second earlier
initial_save_delay = 1  # Initial save delay in seconds
initial_save_complete = False  # Flag to track if initial save delay has passed
gesture_start_time = None  # Store the time when a gesture starts
while True:
    try:
        hand_data = []  # List to store data for each hand

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks and collect_gesture:
            if not gesture_start_time:
                gesture_start_time = time.time()  # Record the time when a gesture starts

            hand_landmarks = results.multi_hand_landmarks[0]  # Assume only one hand is present
            x_ = []
            y_ = []
            data_aux = []

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            hand_data.append(data_aux)

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Backspace gesture
                if predicted_character == 'Backspace':
                    current_time = time.time()
                    if current_time - last_backspace_time > 1:  # 1 second cooldown 
                        word = word[:-1]
                        last_backspace_time = current_time  # Update the last backspace time
                        last_save_time = current_time  # Reset the save timer
                elif predicted_character == 'Reset':
                    word = ''
                    last_save_time = time.time()  # Reset the save timer
                    initial_save_complete = False  # Reset the initial save delay flag
                    gesture_start_time = None  # Reset the gesture start time
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                # Calculate the position to draw the text inside the rectangle
                text_x = x1 + 8  # Adjust as needed to center the text horizontally
                text_y = y1 + 8  # Adjust as needed to center the text vertically

                # Draw the text inside the rectangle
                draw.text((text_x, text_y), predicted_character, font=font, fill=(255, 0, 0))

                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # Check if initial save delay has passed and save after 2 seconds of no change
                current_time = time.time()
                if initial_save_complete:
                    if current_time - last_save_time > 2:
                        last_save_time = current_time
                        word += predicted_character  # Update the word
            else:
                # Reset the initial save time if confidence is below threshold
                last_save_time = time.time()

        else:
            # Reset the gesture start time if no hands are detected
            gesture_start_time = None

        # Check if the initial save delay has passed
        if not initial_save_complete and gesture_start_time:
            current_time = time.time()
            if current_time - gesture_start_time > initial_save_delay:
                initial_save_complete = True

        # Draw the button
        button_rect = (10, 50, 80, 30)  # (x, y, width, height) Adjust width and height to make the button smaller

        button_color = (255, 255, 255)  # Button color
        border_color = (0, 0, 0)        # Border color
        border_thickness = 2            # Border thickness

        # Draw filled rectangle for button
        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), button_color, -1)

        # Draw border rectangle for button
        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), border_color, border_thickness)

        # Add text to the button
        cv2.putText(frame, "Speak", (button_rect[0] + 6, button_rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 1, lineType=cv2.LINE_AA)


        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)

        max_characters_per_line = 47
        line_spacing = 40  # Adjust as needed

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Split the word into lines if it exceeds max_characters_per_line
        lines = [word[i:i+max_characters_per_line] for i in range(0, len(word), max_characters_per_line)]

        # Draw each line
        for i, line in enumerate(lines):
            draw.text((10, 600 + i * line_spacing), line, font=font, fill=(17, 17, 249))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            pass
        elif key == ord('b') or key == ord('B'):
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):
            word = ""
            initial_save_complete = False  # Reset the initial save delay flag
            gesture_start_time = None  # Reset the gesture start time
        elif key == ord('t') or key == ord('T'):
            engine.say(word)
            engine.runAndWait()

        # Reset word if the total number of characters exceeds 61
        if len(word) > 93:
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

"""
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('custom_mode\cust_data\cust_model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    
    0: 'റ',           #1
    1: 'ത',          #2
    2: 'പ',          #3
    3: 'ന',          #4
    4: 'മ',          #5
    5: 'ഴ',          #6
    6: '  ',         #Space  gesture 7
    7: 'Backspace',  # Backspace gesture hi-five 
    8: 'Reset',      #Reset gesture (horn)
}

word = ""
collect_gesture = False

font_path = 'Manjari-Regular.ttf'
font_size = 0.6
font = ImageFont.truetype(font_path, int(font_size * 100))

engine = pyttsx3.init()

# Button properties
button_rect = (10, 50, 100, 40)  # (x, y, width, height)

def on_button_click(event, x, y, flags, param):
    global word
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_rect[0] < x < button_rect[0] + button_rect[2] and button_rect[1] < y < button_rect[1] + button_rect[3]:
            engine.say(word)
            engine.runAndWait()

# Create the window and maximize it
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Set the callback function for mouse events
cv2.setMouseCallback('frame', on_button_click)

# Main loop
last_backspace_time = time.time()
last_save_time = time.time()  # Initialize last save time 1 second earlier
initial_save_delay = 1  # Initial save delay in seconds
initial_save_complete = False  # Flag to track if initial save delay has passed
gesture_start_time = None  # Store the time when a gesture starts
while True:
    try:
        hand_data = []  # List to store data for each hand

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            if not gesture_start_time:
                gesture_start_time = time.time()  # Record the time when a gesture starts

            hand_landmarks = results.multi_hand_landmarks[0]  # Assume only one hand is present
            x_ = []
            y_ = []
            data_aux = []

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            hand_data.append(data_aux)

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                # confidence = model.predict_proba([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Set your threshold here
                # YOUR_THRESHOLD = 0.1

                # Check if confidence is above threshold
                # if confidence[0][int(prediction[0])] > YOUR_THRESHOLD:
                # Backspace gesture
                if predicted_character == 'Backspace':
                    current_time = time.time()
                    if current_time - last_backspace_time > 1:  # 1 second cooldown 
                        word = word[:-1]
                        last_backspace_time = current_time  # Update the last backspace time
                        last_save_time = current_time  # Reset the save timer
                elif predicted_character == 'Reset':
                    word = ''
                    last_save_time = time.time()  # Reset the save timer
                    initial_save_complete = False  # Reset the initial save delay flag
                    gesture_start_time = None  # Reset the gesture start time
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                # Check if initial save delay has passed and save after 2 seconds of no change
                current_time = time.time()
                if initial_save_complete:
                    if current_time - last_save_time > 2:
                        last_save_time = current_time
                        word += predicted_character  # Update the word
            else:
                # Reset the initial save time if confidence is below threshold
                last_save_time = time.time()

        else:
            # Reset the gesture start time if no hands are detected
            gesture_start_time = None

        # Check if the initial save delay has passed
        if not initial_save_complete and gesture_start_time:
            current_time = time.time()
            if current_time - gesture_start_time > initial_save_delay:
                initial_save_complete = True

        # Draw the button
        button_color = (255, 255, 255)  # Button color
        border_color = (0, 0, 0)        # Border color
        border_thickness = 2            # Border thickness

        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), button_color, -1)
        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), border_color, border_thickness)
        cv2.putText(frame, "Speak", (button_rect[0] + 20, button_rect[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)

        max_characters_per_line = 31
        line_spacing = 60  # Adjust as needed

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Split the word into lines if it exceeds max_characters_per_line
        lines = [word[i:i+max_characters_per_line] for i in range(0, len(word), max_characters_per_line)]

        # Draw each line
        for i, line in enumerate(lines):
            draw.text((10, 600 + i * line_spacing), line, font=font, fill=(17, 17, 249))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            pass
        elif key == ord('b') or key == ord('B'):
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):
            word = ""
            initial_save_complete = False  # Reset the initial save delay flag
            gesture_start_time = None  # Reset the gesture start time
        elif key == ord('t') or key == ord('T'):
            engine.say(word)
            engine.runAndWait()

        # Reset word if the total number of characters exceeds 61
        if len(word) > 61:
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
"""