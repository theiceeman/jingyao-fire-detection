# python extract_frames.py

import cv2
import os

def extract_frames(video_path, output_dir, num_frames=100):
    """
    Extract evenly spaced frames from a video.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract (default: 100)
    """
    # create output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # get video ppties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # ensure we don't exceed total_frames
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
        print(f"  Warning: Video has only {total_frames} frames. Extracting all frames.")
    else:
        step = total_frames / num_frames
        frame_indices = [int(i * step) for i in range(num_frames)]
        frame_indices = [min(idx, total_frames - 1) for idx in frame_indices]
    
    # Extract frames
    extracted_count = 0
    for i, frame_idx in enumerate(frame_indices):
        # Set video position to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            frame_filename = os.path.join(output_dir, f"frame_{i+1:03d}.png")
            success = cv2.imwrite(frame_filename, frame)
            if success:
                extracted_count += 1
            else:
                print(f"  Warning: Could not write frame {i+1} to {frame_filename}")
            
            if (i + 1) % 10 == 0:
                print(f"  Extracted {i + 1}/{num_frames} frames...")
        else:
            print(f"  Warning: Could not read frame at index {frame_idx}")
    
    cap.release()
    
    print(f"\nDone! Extracted {extracted_count} frames to '{output_dir}'")

if __name__ == "__main__":
    video_path = "sunsets.mp4"
    output_dir = "data/bcst"
    extract_frames(video_path, output_dir, num_frames=200)