import Image

image = Image.open("smiley.bmp")
pixels = list(image.getdata())
new = []

for p in pixels:
    if p == (255, 0, 0):
        new.append((0, 255, 255))
    else:
        new.append(p)

new_image = Image.new(image.mode, image.size)
new_image.putdata(new)
new_image.save("newsmily.bmp")
