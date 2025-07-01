import os
os.environ['GOOGLE_API_KEY'] = "AIzaSyCIwRrRl9mXBOCShg_oPJWzjlsHQtjXjyA"

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import tempfile
import cv2
from pytubefix import YouTube
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

# Set up the reasoning agent with Gemini 2.0 model
agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Analyzes video content and provides key insights.",
    instructions=[
        "Process uploaded videos or YouTube videos for visual and contextual understanding.",
        "Extract key themes, important topics, and summarize the content.",
        "Identify key objects, faces, and activities if relevant.",
        "Provide a structured analysis of the video.",
        "Use 'DuckDuckGo_Search' to search about the video, if required."
    ],
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Function to process the video and extract frames - NORMAL VIDEO
def process_video(uploaded_file, frame_rate=1):
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())  # Change to read the content as bytes
        temp_path = tmp_file.name
    
    # Use OpenCV to read the video
    cap = cv2.VideoCapture(temp_path)
    
    # Extract frames (let's take 1 frame per second for analysis)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second of the video
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps  # Calculate duration of video in seconds
    
    for sec in range(0, duration, frame_rate):  # Extract frames based on frame_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

    cap.release()
    return frames


def get_youtube_frames(url, frame_rate=1): # YOUTUBE VIDEO
    try:
        # Use Pytube (pytubefix) to extract the YouTube video object
        yt = YouTube(url)
        
        # Get the highest resolution progressive MP4 stream
        video_stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
        
        if not video_stream:
            st.error("No suitable video stream found.")
            return None
        
        # Create a temporary file to save the video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_stream.download(output_path=os.path.dirname(tmp_file.name), filename=os.path.basename(tmp_file.name))
            temp_path = tmp_file.name
            print(f"Downloaded video to: {temp_path}")

        # Now, use OpenCV to read the video and extract frames
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error("Failed to open the video.")
            return None

        frames = []
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get the frames per second
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps  # Calculate the video duration in seconds

        # Extract frames based on frame_rate
        for sec in range(0, duration, frame_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()
        return frames

    except Exception as e:
        st.error(f"An error occurred while processing the YouTube video: {str(e)}")
        return None

# Main application
def main():
    # Streamlit app title
    st.set_page_config(page_title="MultiModal AI", layout="wide")
    st.title("🎥 Multimodal Video Analysis Agent")
    st.markdown("---")

    # Sidebar for configuration and information
    with st.sidebar:
        st.title("⚙️ Configuration")
        frame_rate = st.slider("Frame extraction rate (frames per second)", 1, 10, 1)
        enable_search = st.checkbox("Enable DuckDuckGo Search", value=True)
        
        st.markdown("---")
        st.title("ℹ️ About")
        st.write(
            "This app uses a multimodal AI agent to analyze video content. "
            "You can upload a video or provide a YouTube URL, and the agent will:"
        )
        st.write("- Extract key themes and summarize the content.")
        st.write("- Identify objects, faces, and activities.")
        st.write("- Provide a structured analysis of the video.")
        st.write("- Optionally search the web for additional context.")
        
        st.markdown("---")
        st.warning(
            "⚠️ **Disclaimer**: This tool is for educational and informational purposes only. "
            "All analyses should be reviewed by qualified professionals."
        )

    # Upload video section
    st.header("📤 Upload or Provide a Video")
    st.write(
        "Upload a video file or provide a YouTube URL for analysis. "
        "The AI Agent will analyze the video and provide insights."
    )
    
    # File uploader for video
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    
    # Input for YouTube URL
    youtube_url = st.text_input("Or provide a YouTube video URL")
    
    # Input for dynamic task or question
    task_input = st.text_area("Enter your task or question for the AI Agent:")

    # Button to trigger analysis
    if st.button("Analyze Video & Answer Task"):
        if not task_input:
            st.warning("Please enter a task or question for the AI agent.")
        else:
            with st.spinner("AI is processing... 🧠"):
                try:
                    # Process video
                    frames = None
                    if uploaded_file:
                        frames = process_video(uploaded_file, frame_rate)
                        st.video(uploaded_file)  # Play the uploaded video

                    elif youtube_url:
                        frames = get_youtube_frames(youtube_url, frame_rate)
                        if frames:
                            st.video(youtube_url)  # Play the YouTube video
                        else:
                            st.error("Failed to extract frames from the YouTube video.")
                            return
                    else:
                        st.error("Please upload a video or provide a YouTube URL.")
                        return
                    
                    # Call agent with frames and task input
                    if frames:
                        # Save all frames as temporary images for processing
                        agno_images = []
                        for i, frame in enumerate(frames):
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_img_file:
                                img_path = tmp_img_file.name
                                cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # Save the frame
                                agno_images.append(AgnoImage(filepath=img_path))  # Create AgnoImage object

                        # Call the agent with all frames and task input
                        response = agent.run(task_input, images=agno_images)
                        
                        # Display the AI response
                        st.markdown("### 📋 AI Response:")
                        st.markdown(response.content)

                        # Clean up temporary image files
                        for img in agno_images:
                            if os.path.exists(img.filepath):
                                os.unlink(img.filepath)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Run the Streamlit app
if __name__ == "__main__":
    main()