from setuptools import setup, find_packages
import os

# Read the contents of your requirements.txt file
with open("requirements.txt") as f:
    required = f.read().splitlines()

# Read the contents of your README.md file for the project description
with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

# Define paths for model and test image
model_path = "operate/data/YOLO/models"
test_image_path = "operate/data/YOLO/test_image"
model_file = os.path.join(model_path, "socYOLO.pt")
test_image_file = os.path.join(test_image_path, "image.jpeg")

setup(
    name="self-operating-computer",
    version="1.5.5",
    packages=find_packages(),
    install_requires=required,  # Add dependencies here
    entry_points={
        "console_scripts": [
            "operate=operate.main:main_entry",
        ],
    },
    package_data={
        # Explicitly include the test image and YOLO model
        "operate.data.YOLO.test_image": ["image.jpeg"],
        "operate.data.YOLO.models": ["socYOLO.pt"],
        "operate": ["config.txt"],
    },
    data_files=[
        (model_path, [model_file]),  # Ensure the model file is copied to the correct location
        (test_image_path, [test_image_file]),  # Ensure the test image is copied
    ],
    include_package_data=True,  # Ensures non-code files are included
    long_description=long_description,  # Add project description here
    long_description_content_type="text/markdown",  # Specify Markdown format
)
