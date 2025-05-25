#  Object Dimension Estimation Using a Monocular Camera

##  Project Overview

This project estimates the **depth, width, and height** of objects in an image using a **monocular (single) camera**. It combines real-time object detection and depth estimation through deep learning models. The solution avoids costly setups like LiDAR or stereo cameras by working with just a single camera input.

---

##  Objectives

- Detect objects in real-time or uploaded images.
- Estimate object **depth**, **width**, and **height** using ML models.
- Provide a **user-friendly interface** with result viewing and saving options.
- Use a **PostgreSQL** database for persistent result storage.

---

## Key Features

- **Input Options**:
  - **IP Webcam**: Stream images from a mobile phone using a snapshot URL.  
    *Note: Both the phone (running IP Webcam) and the PC must be on the same Wi-Fi or local network.*
  - **Local Upload**: Upload an image directly from local storage.

- **ML Models Used**:
  - **YOLOv8**: For object detection and bounding box extraction.
  - **GLPN (Godard-Lu-Poggi-Nash)**: For monocular depth estimation.

- **Dimension Calculation**:
  - Estimates **depth** using the GLPN-generated depth map.
  - Calculates **real-world width and height** using bounding box and camera field-of-view geometry.

- **Database Integration**:
  - Uses **PostgreSQL** to save and manage processed results.
  - Includes result viewing and deletion through the Streamlit UI.

- **Visual Outputs**:
  - Displays original and annotated images with bounding boxes and labels.
  - Shows estimated measurements in a clean, readable format.

---

##  Technologies Used

| Component         | Technology                  |
|------------------|-----------------------------|
| UI               | Streamlit                   |
| Object Detection | YOLOv8 (Ultralytics)        |
| Depth Estimation | GLPN (via Hugging Face)     |
| Image Processing | OpenCV, PIL, NumPy          |
| Backend          | Python                      |
| Database         | PostgreSQL                  |

---

## 🚀 How to Run the Project

###  Step 1: Install Python
- Download and install Python 3.9 or higher: [python.org/downloads](https://www.python.org/downloads/)
- Add `python` and `pip` to your system PATH.

###  Step 2: Install Required Libraries
Open a terminal in the project folder and run:
```bash
pip install -r requirements.txt
```

###  Step 3: PostgreSQL Setup
- Download and install PostgreSQL: [postgresql.org/download](https://www.postgresql.org/download/)
- Create a database named `object_db` using pgAdmin or psql:
  ```sql
  CREATE DATABASE object_db;
  ```
- Run the database setup script:
  ```bash
  python database.py
  ```

###  Step 4: Run the Application
Start the Streamlit app:
```bash
streamlit run app.py
```
Visit the app in your browser at:
```
http://localhost:8501
```

---

##  How It Works

1. **User Input**:
   - Enter the IP Webcam URL (`http://<your-ip>:<port>/shot.jpg`)  
     **or**  
   - Upload a local image file.

2. **Processing**:
   - YOLO detects objects and extracts bounding boxes.
   - GLPN generates a depth map.
   - Dimensions are calculated using depth and bounding box data.
   - Models run locally on CPU/GPU.

3. **Results**:
   - Displayed with bounding boxes and measurements.
   - Optionally saved to PostgreSQL for later access.

4. **Saved Data**:
   - Includes processed image and object dimensions.
   - Can be viewed or deleted through the interface.

---

##  Sample Results

![Interface](/Results/ui.png)  
![Bottle-Dimension](/Results/bottle.png)  
![Mouse](/Results/mouse.png)

---

##  Input Source Instructions

To use the IP Webcam feature:
- Install the IP Webcam app on your Android phone.
- Use this format for the snapshot URL:  
  ```
  http://<your-ip>:<port>/shot.jpg
  ```
- Make sure both devices are on the same network.

---

##  Important Notes

1. Only objects detected by YOLO are processed.  
2. This model is calibrated for phone cameras. You can modify parameters to suit other cameras.  
3. Input images should have:
   - A plain background  
   - A single object  
   - Ideal object distance of 51–60 cm from the camera  
   - A steady camera angle  
   - Proper lighting

---

## Project Benefits

- Requires no expensive hardware  
- Works with real-time and offline image inputs  
- Accurate object dimension estimation from a single image  
- Easy-to-use UI and database-backed result management

---

##  Authors

- **Amogh P Lokhande**  
- **Shrivatsa D Desai**  
- **Abhishek Burud**  
- **Umesh L Nayak**  
**Department of Computer Science**  
**Final Year B.Tech Students**

 Contact:  
- shrivatsaddesai@gmail.com  
- amoghlokhande505@gmail.com