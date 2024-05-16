# LOCKED MODE ON RIGHT CLICK AND LEFT CLICK

import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', #retrain needed
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S', 
    19: 'T',
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z',
    26: '  ',         #Space  gesture 7
    27: 'Backspace',  # Backspace gesture hi-five 
    28: 'Reset',      #Reset gesture (devil-horn)
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

''' default camera of current laptop (standby module)
        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (17, 17, 249), 2, cv2.LINE_AA)

        max_characters_per_line = 26
        line_spacing = 60  # Adjust as needed

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Split the word into lines if it exceeds max_characters_per_line
        lines = [word[i:i+max_characters_per_line] for i in range(0, len(word), max_characters_per_line)]

        # Draw each line
        for i, line in enumerate(lines):
            draw.text((10, 360 + i * line_spacing), line, font=font, fill=(17, 17, 249))

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
        if len(word) > 52:
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
'''


"""
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', #retrain needed
    7: 'H', #retrain needed
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S', 
    19: 'T',
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z',
    26: '  ',         #Space  gesture 7
    27: 'Backspace',  # Backspace gesture hi-five 
    28: 'Reset',      #Reset gesture (devil-horn)
    
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



"""


import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', 
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S', 
    19: 'T',
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z',
    26: 'റ',           #1
    27: 'ത',          #2
    28: 'പ',          #3
    29: 'ന',          #4
    30: 'മ',          #5
    31: 'ഴ',          #6
    32: '  ',         #Space  gesture 7
    33: 'Backspace',  # Backspace gesture hi-five 
    34: 'Reset',      #Reset gesture punch(micromax)
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
last_save_time = time.time()
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
            for hand_landmarks in results.multi_hand_landmarks:
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
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    # Automatically save the predicted gesture into "word" after 2 seconds
                    current_time = time.time()
                    if current_time - last_save_time > 2:
                        last_save_time = current_time
                        word += predicted_character  # Update the word
        
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
"""
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A',
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', 
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S', 
    19: 'T',
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z',
    26: 'റ',           #1
    27: 'ത',          #2
    28: 'പ',          #3
    29: 'ന',          #4
    30: 'മ',          #5
    31: 'ഴ',          #6
    32: '  ',         #Space  gesture 7
    33: 'Backspace',  # Backspace gesture hi-five 
    34: 'Reset',      #Reset gesture punch(micromax)
    35: 'horn',        #devil horn
    36: 'horn'
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
last_save_time = time.time()
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
            for hand_landmarks in results.multi_hand_landmarks:
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
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    # Automatically save the predicted gesture into "word" after 2 seconds
                    current_time = time.time()
                    if current_time - last_save_time > 2:
                        last_save_time = current_time
                        word += predicted_character  # Update the word
        
        # Draw the button
        button_color = (255, 255, 255)  # Button color
        border_color = (0, 0, 0)        # Border color
        border_thickness = 2            # Border thickness

        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), button_color, -1)
        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), border_color, border_thickness)
        cv2.putText(frame, "Speak", (button_rect[0] + 20, button_rect[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)

        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 600), word, font=font, fill=(17, 17, 249)) #collected word drawing.....................

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
        elif key == ord('t') or key == ord('T'):
            engine.say(word)
            engine.runAndWait()

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
"""
"""
#multiple hands 
import pickle
import time
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import pyttsx3

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A',
    1: 'B', 
    2: 'C', 
    3: 'D', 
    4: 'E', 
    5: 'F', 
    6: 'G', 
    7: 'H', 
    8: 'I', 
    9: 'J',
    10: 'K', 
    11: 'L', 
    12: 'M', 
    13: 'N', 
    14: 'O', 
    15: 'P', 
    16: 'Q', 
    17: 'R', 
    18: 'S', 
    19: 'T',
    20: 'U', 
    21: 'V', 
    22: 'W', 
    23: 'X', 
    24: 'Y', 
    25: 'Z',
    26: 'റ',           #1
    27: 'ത',          #2
    28: 'പ',          #3
    29: 'ന',          #4
    30: 'മ',          #5
    31: 'ഴ',          #6
    32: '  ',         #Space  gesture 7
    33: 'Backspace',  # Backspace gesture hi-five 
    34: 'Reset',      #Reset gesture punch(micromax)
    35: 'horn'       #devil horn

}

word = ""
collect_gesture = False

font_path = 'Manjari-Regular.ttf'
font_size = 0.6
font = ImageFont.truetype(font_path, int(font_size * 100))

engine = pyttsx3.init()

# Button properties
button_rect = (10, 425, 100, 40)  # (x, y, width, height)

def on_button_click(event, x, y, flags, param):
    global word
    if event == cv2.EVENT_LBUTTONDOWN:
        if button_rect[0] < x < button_rect[0] + button_rect[2] and button_rect[1] < y < button_rect[1] + button_rect[3]:
            engine.say(word)
            engine.runAndWait()

# Create the window
cv2.namedWindow('frame')

# Set the callback function for mouse events
cv2.setMouseCallback('frame', on_button_click)

last_backspace_time = time.time()
last_save_time = time.time()
while True:
    try:
        hand_data = []  # List to store data for each hand

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #this causes the warning Remove this if needed..................................
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #this causes warning Remove this if needed......................................
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    '''
                    #Saving gesture with key press please uncomment the code below with dotted line to use this................................................................................................
                    #Also remember to comment the # Automatically save the predicted gesture section
                    if collect_gesture:
                        word += predicted_character
                        collect_gesture = False
                    '''
                    # Automatically save the predicted gesture into "word" after 2 seconds
                    current_time = time.time()
                    if current_time - last_save_time > 2:
                        last_save_time = current_time
                        #print(f"Saving character: {predicted_character}")
                        word += predicted_character  # Update the word
        
       # Draw the button
        button_color = (255, 255, 255)  # Button color
        border_color = (0, 0, 0)        # Border color
        border_thickness = 2            # Border thickness

        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), button_color, -1)
        cv2.rectangle(frame, (button_rect[0], button_rect[1]), (button_rect[0] + button_rect[2], button_rect[1] + button_rect[3]), border_color, border_thickness)
        cv2.putText(frame, "Speak", (button_rect[0] + 20, button_rect[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)


        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        #cv2.putText(frame, "Press R to Reset", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 330), word, font=font, fill=(17, 17, 249))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            #collect_gesture = True #uncomment this line when needed to enable key press save and the above commented code with dotted line............................................................
            pass
        elif key == ord('b') or key == ord('B'):
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):
            word = ""
        elif key == ord('t') or key == ord('T'):
            engine.say(word)
            engine.runAndWait()

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
"""

'''
#threshold added for more accuracy
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'അ', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'ആ', 27: '  ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

font_path = 'Manjari-Regular.ttf'
font_size = 0.6
font = ImageFont.truetype(font_path, int(font_size * 100))

while True:
    try:
        hand_data = []  # List to store data for each hand

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

                    # Use predict_proba to get class probabilities
                    predicted_probabilities = model.predict_proba([np.asarray(data_aux)])
                    max_probability = np.max(predicted_probabilities)
                    predicted_class = np.argmax(predicted_probabilities)

                    YOUR_THRESHOLD = 0.9
                    if max_probability > YOUR_THRESHOLD:
                        # Draw text using the specified font
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(frame_pil)
                        draw.text((x1, y1 - 10), labels_dict[predicted_class], font=font, fill=(255, 0, 0))
                        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                        if collect_gesture:
                            word += labels_dict[predicted_class]
                            collect_gesture = False  # Reset the flag after collecting the gesture

        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((10, 330), word, font=font, fill=(17, 17, 249))

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            collect_gesture = True
        elif key == ord('b') or key == ord('B'):
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

'''

'''
#single hand

import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: '  ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
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

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                if collect_gesture:
                    word += predicted_character
                    collect_gesture = False  # Reset the flag after collecting the gesture

        # Display the collected word on the frame
        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: " + word, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):    # Press 'q' to quit the video capture window
            break
        elif key == ord('s'):  # Press 's' to start collecting the gesture
            collect_gesture = True
        elif key == ord('b'):  # Press 'b' to backspace
            word = word[:-1]
        elif key == ord('r'):  # Press 'r' to reset completely
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
'''

"""
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up hand detection and drawing tools
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# Define labels dictionary
labels_dict = {
    0: 'A', 1: 'B', 2: 'അ', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'ആ', 27: '  ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

# Initialize variables for storing information
word = ""
collect_gesture = False

# Load font for character visualization
font_path = 'Manjari-Regular.ttf'
font_size = 0.6  # Adjust this value to make the font smaller
font = ImageFont.truetype(font_path, int(font_size * 100))

while True:
    try:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame from camera.")
            break

        H, W, _ = frame.shape

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands in the frame
        results = hands.process(frame_rgb)

        # Process each detected hand separately
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Extract landmark data for the current hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                    # Normalize landmark coordinates relative to hand bounding box
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if data_aux:
                    # Calculate bounding box for the current hand
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    # Predict character using the model
                    prediction = model.predict([np.asarray(data_aux)])
                    predicted_character = labels_dict[int(prediction[0])]

                    # Draw bounding box and prediction around the hand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                    # Convert frame to PIL for text drawing
                    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(frame_pil)
                    draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                    if collect_gesture:
                        word += predicted_character
                        collect_gesture = False  # Reset the flag after collecting the gesture

        # Display the collected word on the frame with adjusted font size
        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        # Convert OpenCV image to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Draw text using the specified font with adjusted size
        draw.text((10, 330),  word, font=font, fill=(17, 17, 249))
        # Convert PIL image back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):    # Press 'q' to quit the video capture window
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to start collecting the gesture
            collect_gesture = True
        elif key == ord('b') or key == ord('B'):  # Press 'b' to backspace
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):  # Press 'r' to reset completely
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0, cv2.CAP_ANY)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'അ', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'ആ', 27: '  ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

# Load Manjari-Regular.ttf font
font_path = 'Manjari-Regular.ttf'
font_size = 0.6  # Adjust this value to make the font smaller
font = ImageFont.truetype(font_path, int(font_size * 100))

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #this causes the warning Remove this if needed
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #this causes warning Remove this if needed 
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
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

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # Draw text using the specified font
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                draw.text((x1, y1 - 10), predicted_character, font=font, fill=(255, 0, 0))
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

                if collect_gesture:
                    word += predicted_character
                    collect_gesture = False  # Reset the flag after collecting the gesture

        # Display the collected word on the frame with adjusted font size
        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        # Convert OpenCV image to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Draw text using the specified font with adjusted size
        #cv2.putText(frame, "Collected Word: ", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        draw.text((10, 330),  word, font=font, fill=(17, 17, 249))
        #draw.text((10, 300), "Collected Word: " + word, font=font, fill=(17, 17, 249))

        # Convert PIL image back to OpenCV format
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == ord('Q'):    # Press 'q' to quit the video capture window
            break
        elif key == ord('s') or key == ord('S'):  # Press 's' to start collecting the gesture
            collect_gesture = True
        elif key == ord('b') or key == ord('B'):  # Press 'b' to backspace
            word = word[:-1]
        elif key == ord('r') or key == ord('R'):  # Press 'r' to reset completely
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
"""



"""

"""
















"""
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'അ', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: 'C', 27: '  ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

# Specify the path to the downloaded Malayalam font file
malayalam_font_path = './Manjari-Regular.ttf'

# Load the Malayalam font
try:
    malayalam_font = ImageFont.truetype(malayalam_font_path, size=30)
except IOError:
    print("Failed to load the Malayalam font. Make sure the font file path is correct.")
    malayalam_font = ImageFont.load_default()

while True:
    try:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
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

            if data_aux:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # Use PIL for text rendering
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)

                if predicted_character.isalpha():  # Check if the character is an alphabet
                    draw.text((x1, y1 - 10), predicted_character, font=malayalam_font, fill=(0, 0, 0))
                else:
                    draw.text((x1, y1 - 10), predicted_character, font=ImageFont.load_default(), fill=(0, 0, 0))

                frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                if collect_gesture:
                    word += predicted_character
                    collect_gesture = False

        cv2.putText(frame, "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2, cv2.LINE_AA)
        cv2.putText(frame, "Collected Word: " + word, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17, 17, 249), 2,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            collect_gesture = True
        elif key == ord('b'):
            word = word[:-1]
        elif key == ord('r'):
            word = ""

    except Exception as e:
        print("An error occurred:", str(e))

print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

"""

"""
import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()

"""

"""
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: '2', 28: '3', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        if data_aux:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
"""
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: ' ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        if data_aux:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            if collect_gesture:
                word += predicted_character
                collect_gesture = False  # Reset the flag after collecting the gesture

    cv2.imshow('frame', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to start collecting the gesture
        collect_gesture = True

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()
"""
"""
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: ' ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        if data_aux:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            if collect_gesture:
                word += predicted_character
                collect_gesture = False  # Reset the flag after collecting the gesture

    # Display the collected word on the frame
    cv2.putText(frame, "Collected Word: " + word, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to start collecting the gesture
        collect_gesture = True

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

"""


"""
import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z',
    26: '1', 27: ' ', 28: ' ', 29: '4', 30: '5', 31: '6', 32: '7', 33: '8', 34: '9', 35: '0'
}

word = ""
collect_gesture = False

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        if data_aux:
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            if collect_gesture:
                word += predicted_character
                collect_gesture = False  # Reset the flag after collecting the gesture

    # Display the collected word on the frame
    cv2.putText(frame , "Press Q to exit this window ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17,249), 2, cv2.LINE_AA)
    cv2.putText(frame , "Press S to save", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17,249), 2, cv2.LINE_AA)
    cv2.putText(frame , "Press B to backspace", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17,249), 2, cv2.LINE_AA)
    cv2.putText(frame , "Press R to Reset", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17,249), 2, cv2.LINE_AA)
    cv2.putText(frame, "Collected Word: " + word, (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (17,17,249), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):    # Press 'q' to quit the video capture window
        break
    elif key == ord('s'):  # Press 's' to start collecting the gesture
        collect_gesture = True
    elif key == ord('b'):  # Press 'b' to backspace
        word = word[:-1]
    elif key == ord('r'):  # Press 'r' to reset completely
        word = ""

# Print the collected word
print("Collected Word:", word)

cap.release()
cv2.destroyAllWindows()

"""








