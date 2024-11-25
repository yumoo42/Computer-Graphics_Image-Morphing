# Computer-Graphics_Image-Morphing
## 1. Environment
Windows 10 / Visual Studio Code / Python 3.11.4\
Python Libraries: cv2, numpy, math, os

## 2. Method Description
### (1) Class Line:
Used to represent feature lines. It can calculate the weight of feature lines during the morphing process and find the corresponding deformation point using calculated u and v values.
- Attributes: Includes the start, end, and midpoint coordinates of the line segment, as well as the length of the line segment and its angle relative to the x-axis. (Two different methods to create line segments).
- Methods: Based on the formulas in the research paper:
  - **perpendicular():** Returns a vector perpendicular to the line segment.
  - **get_u():** Accepts a coordinate point as a parameter and returns the position u of the point on the line segment.
  - **get_v():** Accepts a coordinate point as a parameter and returns the perpendicular distance v of the point from the line segment.
  - **get_point():** Returns a coordinate point based on the u and v values.
  - **get_weight():** Accepts a coordinate point as a parameter and returns the weight of the point, calculated based on its distance to the line segment.

### (2) clip_point(): 
Restricts coordinate points to remain within the bounds of the image.

### (3) gen_warp_line():
Generates intermediate feature lines during the morphing process.\
First, ensure that the angle difference between the source and target feature lines does not exceed π to avoid unnatural rotations during interpolation. Then, use the ratio of the source and target lines during the morphing process to generate intermediate feature lines.

### (4) bilibear():
Performs bilinear interpolation to calculate the pixel values of the new image.\
First, find the integer upper and lower bounds of the coordinate point, ensuring the upper bounds remain within the image range. Then, calculate the distance between the point and the integer values, as well as the values of the four surrounding pixels. Finally, perform bilinear interpolation to calculate the pixel value of the new image.

### (5) warp_point():
Calculates the new position of a coordinate point during the morphing process.\
For each source and target feature line, use gen_warp_line() to generate intermediate feature lines. Then, calculate the new coordinates of the source and target points using the get_point method. Afterward, calculate the weights using get_weight. Finally, compute the resulting source and target points and return the result.

### (6) warp_image():
Performs image morphing.\
First, create three blank images of the same size as the source image. Then, for each pixel in the image, deform it using warp_point, and ensure the new pixel points remain within the image bounds using clip_point. Perform bilinear interpolation to obtain new pixel values for the source and target images. Then, calculate the blended pixel values using the ratio. Finally, update the blank images' pixel values and return the deformed source, target, and blended images.

### (7) onMouse():
Handles mouse events, allowing users to draw lines on the image.\
When the mouse is pressed, record the starting position. As the mouse moves, update the window image to display the drawn lines. When the mouse is released, record the ending position and create a Line object to store in the line_list.

### (8) save_combined_images():
Merges the deformed source image, target image, and blended image into a single image and saves it.

## 3. How to Run the Program
**You need to change the image path in cv2.imread to the path of your own image.**\
You can adjust frame_count to control the number of intermediate images generated and modify DELAY to control the speed of the morphing process.\
After executing main.py, the program will ask whether to use default feature lines or manually draw them. It then calculates the morphed images and stores the results in the result_images list. Finally, the program displays the images in the list and merges the deformed source, target, and result images into a single image for saving.
