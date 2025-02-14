from PIL import Image

image = Image.open("../datasets/UCMerced_LandUse/Images/agricultural/agricultural00.tif")
print(image.size)