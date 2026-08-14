import os
import json
from PIL import Image, ImageDraw

def generate_samples():
    output_dir = os.path.join("c:\\Users\\pooja\\Fruite\\HarvestLenz", "sample_data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create sample_basket_01.jpg (18 fruits)
    img1 = Image.new("RGB", (800, 600), color=(240, 240, 240))
    draw1 = ImageDraw.Draw(img1)
    
    # Draw brown basket
    draw1.rectangle([100, 300, 700, 550], fill=(139, 69, 19), outline=(100, 50, 10), width=5)
    
    # Draw Mangoes (yellow ellipses)
    for i in range(10):
        x = 150 + i * 50
        y = 280 if i % 2 == 0 else 320
        draw1.ellipse([x, y, x+60, y+40], fill=(255, 223, 0), outline=(200, 150, 0))
        
    # Draw Oranges (orange circles)
    for i in range(5):
        x = 200 + i * 70
        y = 360
        draw1.ellipse([x, y, x+50, y+50], fill=(255, 165, 0), outline=(200, 100, 0))
        
    # Draw Grapes (purple circles)
    for i in range(3):
        x = 350 + i * 40
        y = 430
        draw1.ellipse([x, y, x+30, y+30], fill=(128, 0, 128), outline=(80, 0, 80))
        
    img1.save(os.path.join(output_dir, "sample_basket_01.jpg"), "JPEG")
    
    # 2. Create sample_basket_02.jpg (12 fruits)
    img2 = Image.new("RGB", (800, 600), color=(240, 240, 240))
    draw2 = ImageDraw.Draw(img2)
    
    # Draw brown basket
    draw2.rectangle([100, 300, 700, 550], fill=(139, 69, 19), outline=(100, 50, 10), width=5)
    
    # Draw Pomegranates (red circles)
    for i in range(8):
        x = 180 + i * 55
        y = 290 if i % 2 == 0 else 330
        draw2.ellipse([x, y, x+45, y+45], fill=(220, 20, 60), outline=(150, 0, 20))
        
    # Draw Pineapples (criss-cross gold rectangles)
    for i in range(4):
        x = 220 + i * 110
        y = 380
        draw2.rectangle([x, y, x+60, y+80], fill=(218, 165, 32), outline=(150, 100, 0))
        # Draw some green leaves on top
        draw2.polygon([(x+15, y), (x+30, y-30), (x+45, y)], fill=(34, 139, 34))
        
    img2.save(os.path.join(output_dir, "sample_basket_02.jpg"), "JPEG")
    
    # 3. Create sample_response_01.json
    res1 = {
      "scan_id": "c8b4f0b2-ae31-4dbb-871d-15ba17ff5bc3",
      "total_fruits": 18,
      "good": 10,
      "better": 5,
      "medium": 2,
      "reject": 1,
      "average_shelf_life": "6.8 days",
      "fruit_distribution": {
        "Mango": 10,
        "Orange": 5,
        "Grapes": 3
      },
      "markets": {
        "best_market": "Local Mandi",
        "expected_price": "₹1200 - ₹1500 / basket",
        "alternatives": [
          {
            "market": "Export Center",
            "price": "₹2000 - ₹2400 / basket"
          }
        ]
      }
    }
    with open(os.path.join(output_dir, "sample_response_01.json"), "w", encoding="utf-8") as f:
        json.dump(res1, f, indent=2)
        
    # 4. Create sample_response_02.json
    res2 = {
      "scan_id": "df5e8c1b-e538-4b77-9ff5-7aa8e7ff2b38",
      "total_fruits": 12,
      "good": 6,
      "better": 4,
      "medium": 1,
      "reject": 1,
      "average_shelf_life": "9.2 days",
      "fruit_distribution": {
        "Pomegranate": 8,
        "Pineapple": 4
      },
      "markets": {
        "best_market": "Regional Wholesale Mandi",
        "expected_price": "₹1800 - ₹2200 / crate",
        "alternatives": [
          {
            "market": "Retail Supermarket",
            "price": "₹2500 - ₹3000 / crate"
          }
        ]
      }
    }
    with open(os.path.join(output_dir, "sample_response_02.json"), "w", encoding="utf-8") as f:
        json.dump(res2, f, indent=2)

    print("Successfully generated all mock image and json files under sample_data!")

if __name__ == "__main__":
    generate_samples()
