import time
import os
import glob
import requests
from tkinter import filedialog
import tkinter as tk
import logging

log = logging.getLogger(__name__)

API_URL = "http://localhost:8000/api/v1"
USER_ID = "default_user"

def run_search(query):
    """
    Step 3: Perform search via FastAPI backend.
    """
    print(f"Searching for: '{query}'")
    t1 = time.perf_counter()
    
    try:
        response = requests.post(
            f"{API_URL}/search",
            data={"query": query, "user_id": USER_ID, "limit": 5}
        )
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        print(f"Search failed: {e}")
        return

    t2 = time.perf_counter()
    print(f"Time for HTML Search = {t2-t1:.4f}s\n")
    print(f"\n--- Search Results --- (for query='{query}')")
    if not results:
        print("No results found.")
    for idx, res in enumerate(results):
        print(
            f"{idx+1}. [{res.get('filename', 'N/A')}] "
            f"Time: {res.get('timestamp', 0):.2f}s | "
            f"Frame: {res.get('frame_idx', 'N/A')} | "
            f"Score: {res.get('score', 0):.4f} | "
            f"Video ID: {res.get('video_id', 'N/A')}"
        )
    print("----------------------\n")


def process_video(vid_path):
    import datetime
    print(f"Uploading {os.path.basename(vid_path)}...")
    try:
        t1 = datetime.datetime.now()   # FIX: was datetime.datetime.time (class ref, not a call)
        with open(vid_path, 'rb') as f:
            response = requests.post(
                f"{API_URL}/upload",
                data={"user_id": USER_ID},
                files={"file": (os.path.basename(vid_path), f, "video/mp4")}
            )
        response.raise_for_status()
        task_info = response.json()
        task_id = task_info.get("task_id")
        print(f"Started processing. Task ID: {task_id}")
        
        while True:
            time.sleep(2)
            try:
                status_res = requests.get(f"{API_URL}/status/{task_id}")
                if status_res.status_code == 200:
                    status = status_res.json().get("status")
                    if status == "completed":
                        print(f"[{os.path.basename(vid_path)}] Processing completed!")
                        break
                    elif status == "failed":
                        print(f"[{os.path.basename(vid_path)}] Processing failed: {status_res.json().get('error')}")
                        break
                    else:
                        print(f"[{os.path.basename(vid_path)}] Status: {status}...")
                else:
                    print(f"Error checking status: {status_res.status_code}")
                    break
            except Exception as e:
                print(f"Status check failed: {e}")
                break
        t2 = datetime.datetime.now()   # FIX: was datetime.datetime.time (class ref, not a call)
        print(f"\n\nTotal time required : {(t2-t1).total_seconds():.2f}s\n")
    except Exception as e:
        print(f"Failed to upload {vid_path}: {e}")

def add_videos_flow():
    # Hide the main tkinter window
    root = tk.Tk()
    root.withdraw()

    choice = input("Select Source - Do you want to add a whole FOLDER? (y/n)\n : ")
    
    video_files = []
    
    if choice.lower() == 'y':
        folder = filedialog.askdirectory(title="Select Video Folder")
        if not folder:
            return
        video_files = glob.glob(os.path.join(folder, "*.mp4"))
        if not video_files:
            print("Error: No .mp4 files found in that folder.")
            return
    elif choice.lower() == 'n' : 
        files = filedialog.askopenfilenames(title="Select Video Files", filetypes=[("MP4 Videos", "*.mp4")])
        if not files: return
        video_files = list(files)
    else :
        print("Invalid input")
        return

    print(f"Selected {len(video_files)} videos. Uploading to backend...")
    
    for vid_path in video_files:
        process_video(vid_path)
    
    print("Finished uploading videos.")
    return
    

def remove_videos_flow():
    print("Warning: Remove videos functionality is not yet exposed by the backend API.")

def main():
    print("AI Video Search Engine Client Started.")
    print(f"Ensure the API is running at {API_URL}")

    while True:        
        choice = input("\n1.Search\n2.Add Videos\n3.Remove videos\n4.Exit\n : ")
        try :
            choice = int(choice)
        except :
            continue
        if choice == 1 :
             query = input("Enter search query : ")
             if query:
                run_search(query)
                input("\nPress Enter to continue...")
                 
        elif choice == 2:
            add_videos_flow()
            
        elif choice == 3:
            remove_videos_flow()
            
        elif choice == 4 or choice == "":
            break
            
    print("Exited\n")

if __name__ == "__main__":
    main()
