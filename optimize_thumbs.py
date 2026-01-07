import os
from pathlib import Path
from PIL import Image  # pip install Pillow

# Configuration
BASE_DIR = Path(__file__).resolve().parent
THUMBS_DIR = BASE_DIR / "static" / "uploads" / "thumbs"
MAX_SIZE = (600, 600)
QUALITY = 85

def get_size_mb(path):
    return path.stat().st_size / (1024 * 1024)

def optimize_library():
    if not THUMBS_DIR.exists():
        print(f"Error: Directory not found at {THUMBS_DIR}")
        return

    print(f"--- Starting Optimization in {THUMBS_DIR.name} ---")
    
    files = [f for f in THUMBS_DIR.iterdir() if f.is_file() and f.name != ".gitkeep"]
    total_files = len(files)
    
    start_size = sum(f.stat().st_size for f in files)
    processed_count = 0
    skipped_count = 0
    
    print(f"Found {total_files} files. Total size: {start_size / (1024*1024):.2f} MB")

    for i, file_path in enumerate(files, 1):
        try:
            ext = file_path.suffix.lower()
            
            # Skip non-images or GIFs (processing GIFs often kills animation)
            if ext not in ['.jpg', '.jpeg', '.png', '.webp']:
                skipped_count += 1
                continue
            
            if ext == '.gif':
                print(f"[{i}/{total_files}] Skipping GIF: {file_path.name}")
                skipped_count += 1
                continue

            # Open and Process
            original_size = file_path.stat().st_size
            
            with Image.open(file_path) as img:
                # Check if resize is actually needed
                if img.width <= MAX_SIZE[0] and img.height <= MAX_SIZE[1]:
                    # Even if size is okay, we re-save to apply optimization
                    pass
                
                img.thumbnail(MAX_SIZE)
                
                # Save logic
                if ext in ['.jpg', '.jpeg']:
                    # Convert to RGB in case it was CMYK or RGBA
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(file_path, optimize=True, quality=QUALITY)
                    
                elif ext == '.png':
                    img.save(file_path, optimize=True)
                    
                elif ext == '.webp':
                    img.save(file_path, optimize=True, quality=QUALITY)

            new_size = file_path.stat().st_size
            reduction = (original_size - new_size) / 1024
            print(f"[{i}/{total_files}] Optimized {file_path.name} (-{reduction:.1f} KB)")
            processed_count += 1

        except Exception as e:
            print(f"[{i}/{total_files}] ERROR processing {file_path.name}: {e}")
            skipped_count += 1

    end_size = sum(f.stat().st_size for f in THUMBS_DIR.iterdir() if f.is_file())
    saved_mb = (start_size - end_size) / (1024 * 1024)

    print("\n--- Summary ---")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Space Saved: {saved_mb:.2f} MB")
    print("Done.")

if __name__ == "__main__":
    optimize_library()