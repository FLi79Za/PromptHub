import sqlite3
import os
from pathlib import Path
from PIL import Image

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "prompts.db"
UPLOAD_DIR = BASE_DIR / "static"

def optimize_gifs_v2():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Find prompts that still point to .gif files
    rows = conn.execute("SELECT id, title, thumbnail FROM prompts WHERE thumbnail LIKE '%.gif'").fetchall()
    
    print(f"Found {len(rows)} GIFs to optimize...")
    
    for row in rows:
        old_rel_path = row["thumbnail"]
        full_old_path = UPLOAD_DIR / old_rel_path
        
        if not full_old_path.exists():
            print(f"Skipping {row['id']}: File not found ({old_rel_path})")
            continue
            
        print(f"Processing: {row['title']}...")
        
        try:
            img = Image.open(full_old_path)
            
            # Setup new filename
            new_rel_path = old_rel_path.rsplit('.', 1)[0] + ".webp"
            full_new_path = UPLOAD_DIR / new_rel_path
            
            # --- ROBUST ANIMATION EXTRACTION ---
            frames = []
            durations = []
            
            try:
                while True:
                    # Limit frames to avoid memory crashes on massive GIFs
                    if len(frames) > 60: break
                    
                    durations.append(img.info.get('duration', 100))
                    
                    # Convert current frame
                    f = img.convert("RGBA")
                    f.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    frames.append(f)
                    
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            if frames:
                frames[0].save(
                    full_new_path,
                    format="WEBP",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=durations, # Use the list of durations we captured
                    loop=0,
                    quality=80
                )
                
                # Update DB
                conn.execute("UPDATE prompts SET thumbnail = ? WHERE id = ?", (new_rel_path, row["id"]))
                conn.commit()
                
                # Close and Remove Old
                img.close()
                os.remove(full_old_path)
                print(f" -> Success! Converted to {new_rel_path}")
            else:
                print(" -> Error: No frames extracted.")

        except Exception as e:
            print(f" -> Failed: {e}")

    conn.close()
    print("\nDone.")

if __name__ == "__main__":
    optimize_gifs_v2()