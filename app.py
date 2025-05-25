
import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import streamlit as st

from overkillpone import process_image  # Assuming this is the module for your YOLO model
import database

# Initialize database with error handling
try:
    database.initialize_table()
except Exception as e:
    st.error(f"❌ Failed to initialize database: {e}")

st.set_page_config(page_title="Project Final Year", layout="wide")
st.title("Object Dimension Measurement")

page = st.sidebar.selectbox("Choose Page", ["Detect Objects", "View Saved Results"])

if page == "Detect Objects":
    option = st.radio("Choose Input Source:", [
        "IP Webcam",
      
        "Local Image Upload"
    ])

    if option == "IP Webcam":
        st.header("📱 IP Webcam")
        # http://192.168.29.79:8001/shot.jpg
        ip_url = st.text_input("Enter IP Webcam Snapshot URL", value="/shot.jpg")
        if st.button("📸 Capture from IP Webcam"):
            try:
                response = requests.get(ip_url, timeout=5)
                response.raise_for_status()
                img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = Image.open(BytesIO(img_array)).convert("RGB")
                st.image(frame, caption="Captured Frame", width=500)

                with st.spinner("Processing..."):
                    try:
                        output_image, depths, widths, heights, classes = process_image(frame)
                        if not classes:
                                st.warning("⚠️ No objects detected. Please try capturing the image again.")
                                st.warning("Keep Steady angle, Check proper Light")
                                st.stop()
                    except Exception as e:
                        st.error(f"Error during image processing: {e}")
                        output_image, classes = None, []

                if output_image:
                    st.image(output_image, caption="Detection Result", width=500)
                    st.success("Processing Done ✅")

                    detections_list = []
                    for i, cls in enumerate(classes):
                        st.write(f" Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
                        detections_list.append(f" Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

                    st.session_state["output_image"] = output_image
                    st.session_state["detections_list"] = detections_list
            except Exception as e:
                st.error(f"❌ Error: {e}")



    # Paste if Droid


    elif option == "Local Image Upload":
        st.header("Upload Local Image")
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if file:
            try:
                img = Image.open(file).convert("RGB")
                st.image(img, caption="Uploaded Image", width=500)

                if st.button("Process Uploaded Image"):
                    with st.spinner("Processing..."):
                        try:
                            output_image, depths, widths, heights, classes = process_image(img)
                            if not classes:
                                st.warning("⚠️ No objects detected. Please try capturing the image again.")
                                st.warning("Keep Steady angle, Check proper Light")
                                st.stop()
                        except Exception as e:
                            st.error(f"Error during image processing: {e}")
                            output_image, classes = None, []

                    if output_image:
                        st.image(output_image, caption="Detection Result", width=500)
                        st.success("Processing Done ✅")

                        detections_list = []
                        for i, cls in enumerate(classes):
                            st.write(f" Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")
                            detections_list.append(f" Depth={depths[i]:.2f}, Width={widths[i]:.2f}, Height={heights[i]:.2f}")

                        st.session_state["output_image"] = output_image
                        st.session_state["detections_list"] = detections_list
            except Exception as e:
                st.error(f"❌ Error processing uploaded image: {e}")

    if "output_image" in st.session_state and "detections_list" in st.session_state:
        if st.button(" Save to Database"):
            try:
                database.save_result_to_db(
                    st.session_state["output_image"],
                    st.session_state["detections_list"]
                )
                st.success("✅ Saved to database.")
            except Exception as e:
                st.error(f"❌ Failed to save to database: {e}")

elif page == "View Saved Results":
    st.header("📂 Saved Results")
    try:
        results = database.load_saved_results()
        if results:
            for row in results:
                try:
                    img_bytes = row['image']
                    detections = row['detections']
                    timestamp = row['timestamp']
                    result_id = row['id']

                    img_pil = Image.open(BytesIO(img_bytes))
                    time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else "Unknown"

                    st.image(img_pil, caption=f"Saved on {time_str}", width=500)
                    st.write("Detections:")
                    st.json(eval(detections))

                    if st.button(f"🗑️ Delete {result_id}", key=f"delete_{result_id}"):
                        try:
                            database.delete_result(result_id)
                            st.success("Deleted. Refresh to update.")
                        except Exception as e:
                            st.error(f"❌ Failed to delete record: {e}")
                except Exception as e:
                    st.error(f"❌ Failed to render a saved result: {e}")
        else:
            st.info("No saved results yet.")
    except Exception as e:
        st.error(f"❌ Failed to load saved results: {e}")

