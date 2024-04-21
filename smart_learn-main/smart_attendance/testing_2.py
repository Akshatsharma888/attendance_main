''' directory to create named images folder.'''

import os
import shutil

# Directory where the known faces and test images are located
image_directory = "smart_attendance"

# New directory for images with names
new_directory = "named_images"

# Create new directory if it doesn't exist
if not os.path.exists(new_directory):
    os.makedirs(new_directory)

# Get a list of all image files in the directory
image_files = [f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Copy each image to the new directory with the name of the person as the filename
for image_file in image_files:
    name = os.path.splitext(image_file)[0]  # Name of the person is the filename without the extension
    src_path = os.path.join(image_directory, image_file)
    dst_path = os.path.join(new_directory, name + os.path.splitext(image_file)[1])
    shutil.copy2(src_path, dst_path)