from PIL import Image, ImageDraw, ImageFont

# ======================
# Create Canvas
# ======================
W, H = 900, 900
img = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(img)

# ======================
# Fonts (Windows)
# ======================
font_path = r"C:\Windows\Fonts\arialbd.ttf"

try:
    title_font = ImageFont.truetype(font_path, 44)
    box_font = ImageFont.truetype(font_path, 36)
    arrow_font = ImageFont.truetype(font_path, 44)
except:
    title_font = ImageFont.load_default()
    box_font = ImageFont.load_default()
    arrow_font = ImageFont.load_default()

# ======================
# Colors
# ======================
navy = (0, 55, 120)
green = (20, 130, 70)
orange = (230, 120, 30)
purple = (105, 70, 170)

light_green = (232, 248, 238)
light_blue = (232, 241, 252)
light_orange = (255, 242, 225)
light_purple = (242, 237, 252)

# ======================
# Title
# ======================
draw.text(
    (50, 30),
    "Grid-Strength Challenge Chain",
    fill=navy,
    font=title_font
)

# ======================
# Layout Parameters
# ======================
box_w = 620
box_h = 100
x = (W - box_w) // 2

# Y positions
ys = [130, 300, 470, 640]

# ======================
# Box Definitions
# ======================
# Box Definitions
# ======================

boxes = [
    ("Renewable Integration", green, light_green),
    ("Converter-Based Resources", navy, light_blue),
    ("Grid Strength Conditions", orange, light_orange),
    ("Voltage & Stability Challenges", purple, light_purple)
]

# ======================
# Draw Boxes
# ======================
for i, (text, outline_color, fill_color) in enumerate(boxes):

    y = ys[i]

    draw.rounded_rectangle(
        [x, y, x + box_w, y + box_h],
        radius=20,
        fill=fill_color,
        outline=outline_color,
        width=4
    )

    bbox = draw.textbbox((0, 0), text, font=box_font)

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    draw.text(
        (
            x + (box_w - text_w) / 2,
            y + (box_h - text_h) / 2
        ),
        text,
        fill=outline_color,
        font=box_font
    )

    # Draw arrow between boxes
    if i < len(boxes) - 1:

        arrow_text = "↓"

        bbox = draw.textbbox(
            (0, 0),
            arrow_text,
            font=arrow_font
        )

        arrow_w = bbox[2] - bbox[0]

        draw.text(
            (
                W/2 - arrow_w/2,
                y + box_h + 15
            ),
            arrow_text,
            fill=navy,
            font=arrow_font
        )

# ======================
# Save Image
# ======================
output_file = "grid_strength_challenge_chain.png"

img.save(output_file)

print(f"Saved: {output_file}")

# Show image
img.show()