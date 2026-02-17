from PIL import Image
import os

def crop_banner(input_path, output_path):
    img = Image.open(input_path)
    width, height = img.size
    
    # Define the crop box (left, top, right, bottom)
    # We want a 3:1 or 4:1 aspect ratio centered vertically
    target_height = width // 3
    top = (height - target_height) // 2
    bottom = top + target_height
    
    left = 0
    right = width
    
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(output_path)
    print(f"Banner cropped to {img_cropped.size} and saved to {output_path}")

if __name__ == "__main__":
    input_file = "/home/fred/.gemini/antigravity/brain/c74aa91c-5c5b-4175-a65e-a5da70d9df8f/alita_g_banner_ribbon_v4_1771325079927.png"
    output_file = "assets/banner.png"
    crop_banner(input_file, output_file)
