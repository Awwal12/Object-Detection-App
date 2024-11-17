import cv2
import numpy as np
import sys
import os
import time
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLO model and COCO class names

net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get YOLO output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Function to detect objects in a frame


def detect_objects(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids, confidences, boxes = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i] % len(colors)]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidences[i]:.2f}", (
                x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    return frame

# Function for detecting objects in an image


# Function for detecting objects in an image
def detect_objects_in_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return

    img = detect_objects(img)

    # Create a resizable window
    cv2.namedWindow("Image Object Detection", cv2.WINDOW_NORMAL)

    # Optional: Set a default window size (you can adjust this)
    cv2.resizeWindow("Image Object Detection", 800, 600)

    # Display the image
    cv2.imshow("Image Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Function for detecting objects in real-time using webcam


def detect_objects_in_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    starting_time = time.time()
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame_id += 1
        frame = detect_objects(frame)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.imshow("Real-Time Object Detection", frame)

        if cv2.waitKey(1) == 27:  # Press 'ESC' to exit
            print("ESC pressed, exiting...")
            break
    # Window closed
        if cv2.getWindowProperty("Real-Time Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed, exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# Create the main window
app = ctk.CTk()
app.geometry("1100x450")
app.title("Object Detection App")
ctk.set_appearance_mode('Dark')
app.resizable(False, False)


def get_image_path(image_name):
    return os.path.join(os.path.dirname(__file__), "Ui Images", image_name)

# Load images with error handling


def load_image(image_path, size):
    try:
        return ctk.CTkImage(Image.open(image_path), size=size)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Clear main frame


def clear_main_frame():
    for widget in main_frame.winfo_children():
        widget.destroy()

# Handle launch detection button click


def launch_application():
    clear_main_frame()
    mode = ctk.CTkOptionMenu(
        main_frame, values=["Image/Video Mode", "Webcam Mode"], command=start_detection)
    mode.grid(row=0, column=0, padx=20, pady=20)

# Start detection based on selected mode


def start_detection(selected_mode):
    if selected_mode == "Image/Video Mode":
        image_path = filedialog.askopenfilename(title="Select Image/video", filetypes=[
                                                ("Image/Video Files", "*.jpg;*.png;*.jpeg;*.webp;*.mkv;*.mp4;")])
        if image_path:
            detect_objects_in_image(image_path)
    elif selected_mode == "Webcam Mode":
        detect_objects_in_webcam()

# About us section


def show_about_us():
    clear_main_frame()
    about_us_text = ("Welcome to the Object Detection App!\n\n"
                     "This project demonstrates real-time object detection using YOLOv7 with image and webcam modes.")
    label = ctk.CTkLabel(main_frame, text=about_us_text,
                         font=ctk.CTkFont(size=15), anchor="w", justify="left")
    label.grid(row=0, column=0, padx=20, pady=20)


launch_icon = load_image(get_image_path(
    "C:\\Users\\MY PC\\Documents\\fyp\\tap.png"), size=(40, 40))
image_icon = load_image(get_image_path(
    "C:\\Users\MY PC\Documents\\fyp\\house.png"), size=(20, 20))
how_to_use_icon = load_image(get_image_path(
    "C:\\Users\\INEWTON\\Desktop\\Ui Images\\Main\\user-guide1.png"), size=(20, 20))
about_us_icon = load_image(get_image_path(
    "C:\\Users\\MY PC\\Documents\\fyp\\about.png"), size=(20, 20))


def show_home():
    clear_main_frame()

    # Create a welcome label
    label = ctk.CTkLabel(main_frame, text="Welcome to the Object Detection App!",
                         font=ctk.CTkFont(size=20, weight="bold"))
    label.grid(row=0, column=0, padx=(230, 0), pady=(20, 10))

    # Create the "Launch Application" button with an image
    launch_button = ctk.CTkButton(main_frame, width=500, height=100, text="Click To Start",
                                  image=launch_icon, font=ctk.CTkFont(size=25), compound="left", command=launch_application)
    launch_button.grid(row=1, column=0, padx=(230, 0), pady=(80, 10))


# Sidebar configuration
sidebar_frame = ctk.CTkFrame(app, width=200, corner_radius=0)
sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")

home_button = ctk.CTkButton(
    sidebar_frame, text="Home", image=image_icon, command=show_home)
home_button.grid(row=0, column=0, padx=20, pady=(30, 10))

about_us_button = ctk.CTkButton(
    sidebar_frame, text="About Project", image=about_us_icon, command=show_about_us)
about_us_button.grid(row=1, column=0, padx=20, pady=(30, 10))

system_dropdown = ctk.CTkOptionMenu(
    sidebar_frame, values=["Dark", "Light"], command=ctk.set_appearance_mode)
system_dropdown.grid(row=2, column=0, padx=20, pady=(280, 10))

main_frame = ctk.CTkFrame(app)
main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)

# Run the app
app.mainloop()
