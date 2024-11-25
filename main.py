import cv2
import numpy as np
import math
import os

# Constants
WINDOW_X = 100
WINDOW_Y = 200
PADDING = 20
DELAY = 800
param_a = 1
param_b = 2
param_p = 0
fram_count = 10

# Global variable
source_line_list = []
destination_line_list = []
destination2_line_list = []
warp_line_list = []

class Line:
    def __init__(self, start_point = None, end_point = None, middle = None, length = None, degree = None):
        if (start_point is not None and end_point is not None):
            self.P = np.array(start_point)
            self.Q = np.array(end_point)
            self.M = (self.P + self.Q) / 2
            self.diff = self.Q - self.P
            self.length = np.linalg.norm(self.Q - self.P)
            self.degree = math.atan2(self.diff[1], self.diff[0])
        elif (middle is not None and length is not None and degree is not None):
            self.M = np.array(middle)
            deltaX = length / 2 * math.cos(degree)
            deltaY = length / 2 * math.sin(degree)
            self.P = np.array([self.M[0] - deltaX, self.M[1] - deltaY])
            self.Q = np.array([self.M[0] + deltaX, self.M[1] + deltaY])
            self.diff = self.Q - self.P
            self.length = length
            self.degree = degree

    def perpendicular(self):
        return np.array([self.diff[1], -self.diff[0]])

    def get_u(self, X):
        return np.dot((X - self.P), (self.Q - self.P)) / (self.length * self.length)
    
    def get_v(self, X):
        return np.dot((X - self.P), self.perpendicular()) / self.length
    
    def get_point(self, u, v):
        return self.P + u * (self.Q - self.P) + (v * self.perpendicular()) / self.length
    
    def get_weight(self, X):
        u = self.get_u(X)
        if (u > 1):
            dist = np.linalg.norm(self.Q - X)
        elif (u < 0):
            dist = np.linalg.norm(self.P - X)
        else:
            dist = abs(self.get_v(X))
        return pow((pow(self.length, param_p) / (param_a + dist)), param_b)

def clip_point(point, height, width):
    out = point.copy()
    if out[0] < 0:
        out[0] = 0
    elif out[0] >= width:
        out[0] = width - 1
    if out[1] < 0:
        out[1] = 0
    elif out[1] >= height:
        out[1] = height - 1
    return out

def gen_warp_line(source_line, destination_line, ratio):
    while source_line.degree - destination_line.degree > np.pi:
        destination_line.degree += np.pi
    while destination_line.degree - source_line.degree > np.pi:
        source_line.degree += np.pi
    M = (1 - ratio) * source_line.M + ratio * destination_line.M
    length = (1 - ratio) * source_line.length + ratio * destination_line.length
    degree = (1 - ratio) * source_line.degree + ratio * destination_line.degree
    return Line(middle=M, length=length, degree=degree)

def bilinear(img, point):
    x_floor = int(np.floor(point[0]))
    y_floor = int(np.floor(point[1]))
    x_ceil = int(np.ceil(point[0]))
    y_ceil = int(np.ceil(point[1]))
    a = point[0] - x_floor
    b = point[1] - y_floor
    x_ceil = min(x_ceil, img.shape[1] - 1)
    y_ceil = min(y_ceil, img.shape[0] - 1)
    top_left = img[y_floor, x_floor]
    top_right = img[y_floor, x_ceil]
    bottom_left = img[y_ceil, x_floor]
    bottom_right = img[y_ceil, x_ceil]
    out = (1 - b) * ((1 - a) * top_left + a * top_right) + b * ((1 - a) * bottom_left + a * bottom_right)
    return out

def warp_point(point, src_line_list, dest_line_list, ratio):
    source_point_sum = np.array([0.0, 0.0])
    destination_point_sum = np.array([0.0, 0.0])
    source_weight_sum = 0
    destination_weight_sum = 0
    for i in range(len(src_line_list)):
        source_line = src_line_list[i]
        destination_line = dest_line_list[i]
        warp_line = gen_warp_line(source_line, destination_line, ratio)
        source_point = source_line.get_point(warp_line.get_u(point), warp_line.get_v(point))
        destination_point = destination_line.get_point(warp_line.get_u(point), warp_line.get_v(point))
        source_weight = source_line.get_weight(source_point)
        destination_weight = destination_line.get_weight(destination_point)
        source_point_sum += source_point * source_weight
        destination_point_sum += destination_point * destination_weight
        source_weight_sum += source_weight
        destination_weight_sum += destination_weight
    src_point = source_point_sum / source_weight_sum
    dest_point = destination_point_sum / destination_weight_sum
    return src_point, dest_point

def warp_image(source_img, destination_img, src_line_list, dest_line_list, ratio):
    source = source_img.copy()
    dest = destination_img.copy()
    emptyImage = np.zeros_like(source)
    emptyImage_right = np.zeros_like(source)
    emptyImage_left = np.zeros_like(source)
    for i in range(emptyImage.shape[1]):
        for j in range(emptyImage.shape[0]):
            point = np.array([i, j])
            src_point, dest_point = warp_point(point, src_line_list, dest_line_list, ratio)
            src_point = clip_point(src_point, emptyImage.shape[0], emptyImage.shape[1])
            dest_point = clip_point(dest_point, emptyImage.shape[0], emptyImage.shape[1])
            source_scalar = bilinear(source, src_point)
            destination_scalar = bilinear(dest, dest_point)
            scalar = (1 - ratio) * source_scalar + ratio * destination_scalar
            emptyImage[j, i] = scalar
            emptyImage_right[j, i] = source_scalar
            emptyImage_left[j, i] = destination_scalar
    return emptyImage, emptyImage_right, emptyImage_left

start_point = np.array([0.0, 0.0])
end_point = np.array([0.0, 0.0])
def onMouse(event, x, y, flags, param):
    global start_point, end_point
    line_list, window_name, image = param
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        print("start:", start_point)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        img =  image.copy()
        end_point = (x, y)
        cv2.arrowedLine(img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow(window_name, img)
    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        print("end:", end_point)
        cv2.arrowedLine(image, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow(window_name, image)
        line = Line(start_point=start_point, end_point=end_point)
        line_list.append(line)

def save_combined_images(index, warped_source_image, warped_destination_image, result_image, output_dir="output_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    combined_image = np.hstack((warped_source_image, warped_destination_image, result_image))
    filename = f"{output_dir}/image_{index}.png"
    cv2.imwrite(filename, combined_image)

# Read images
image_source = cv2.imread("images\women.jpg")
image_dest = cv2.imread("images\cheetah.jpg")
image_dest2 = cv2.imread("images\Putin.jpg")

# Check image dimensions
if not (image_source.shape == image_dest.shape == image_dest2.shape):
    image_dest = cv2.resize(image_dest, (image_source.shape[1], image_source.shape[0]))
    image_dest2 = cv2.resize(image_dest2, (image_source.shape[1], image_source.shape[0]))

# Create images for showing
showImageSource = image_source.copy()
showImageDest = image_dest.copy()
showImageDest2 = image_dest2.copy()

# Create windows
cv2.namedWindow("Source Image")
cv2.namedWindow("Destination Image")
cv2.namedWindow("Destination2 Image")
cv2.moveWindow("Source Image", WINDOW_X, WINDOW_Y)
cv2.moveWindow("Destination Image", WINDOW_X + PADDING + image_source.shape[1], WINDOW_Y)
cv2.moveWindow("Destination2 Image", WINDOW_X + 2*PADDING + 2*image_source.shape[1], WINDOW_Y)
cv2.imshow("Source Image", showImageSource)
cv2.imshow("Destination Image", showImageDest)
cv2.imshow("Destination2 Image", showImageDest2)

if __name__ == "__main__":
    print("Do you want to use predefined lines? (Y/N):\n")
    # while True:
    key = cv2.waitKey(0) & 0xFF
    if key == ord("y"):
        print("Using predefined lines.")
        source = [[(115, 43), (114, 101)], [(61, 42), (96, 47)], [(168, 42), (136, 46)], [(93, 131), (138, 129)], [(37, 81), (68, 156)], [(190, 79), (163, 153)]]
        destination = [[(126, 28), (128, 137)], [(35, 17), (76, 27)], [(226, 14), (179, 26)], [(101, 178), (169, 178)], [(4, 85), (46, 161)], [(253, 89), (206, 162)]]
        destination2 = [[(131, 82), (129, 134)], [(87, 85), (113, 87)], [(178, 86), (154, 88)], [(111, 158), (156, 157)], [(54, 95), (73, 162)], [(204, 99), (191, 162)]]
        source = np.array(source)
        destination = np.array(destination)
        destination2 = np.array(destination2)
        for i in range(len(source)):
            l_P = np.array(source[i][0])
            l_Q = np.array(source[i][1])
            t_P = np.array(destination[i][0])
            t_Q = np.array(destination[i][1])
            r_P = np.array(destination2[i][0])
            r_Q = np.array(destination2[i][1])
            l = Line(start_point=l_P, end_point=l_Q)
            t = Line(start_point=t_P, end_point=t_Q)      
            r = Line(start_point=r_P, end_point=r_Q)        
            source_line_list.append(l)
            destination_line_list.append(t)
            destination2_line_list.append(r)
        print("Start Computing\n")
        
    elif key == ord("n"):     
        print("Please enter [a] to draw line\n")
        print("If you want to continue drawing lines, please press the 'a' key. If not, please press the 'q' key.\n")
        # Set mouse callbacks
        cv2.setMouseCallback("Source Image", onMouse, (source_line_list, "Source Image", showImageSource))
        cv2.setMouseCallback("Destination Image", onMouse, (destination_line_list, "Destination Image", showImageDest))
        cv2.setMouseCallback("Destination2 Image", onMouse, (destination2_line_list, "Destination2 Image", showImageDest2))

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27 or key == ord('q'):
                print("Start Computing\n")
                break
            elif key == ord('a'):
                print("Please draw the line of three pictures.\n")

    result_images = []
    for i in range(fram_count):
        ratio = i / fram_count
        result_images.append(warp_image(image_source, image_dest, source_line_list, destination_line_list, ratio))
        print("loading..." + str(i) + " " + str(fram_count))

    for i in range(1, fram_count + 1):
        ratio = i / fram_count
        result_images.append(warp_image(image_dest, image_dest2, destination_line_list, destination2_line_list, ratio))
        print("loading..." + str(i) + " " + str(fram_count))


    cv2.namedWindow("Result Image")
    cv2.moveWindow("Result Image", WINDOW_X + 3 * PADDING + 3 * image_source.shape[1], WINDOW_Y + image_source.shape[0] + PADDING)
    cv2.namedWindow("Warped Source Image")
    cv2.moveWindow("Warped Source Image", WINDOW_X, WINDOW_Y + image_source.shape[0] + PADDING)
    cv2.namedWindow("Warped Destination Image")
    cv2.moveWindow("Warped Destination Image", WINDOW_X + PADDING + image_source.shape[1], WINDOW_Y + image_source.shape[0] + PADDING)
    cv2.namedWindow("Warped Destination2 Image")
    cv2.moveWindow("Warped Destination2 Image", WINDOW_X + 2 * PADDING + 2 * image_source.shape[1], WINDOW_Y + image_source.shape[0] + PADDING)
    for i in range(len(result_images)):
        cv2.imshow("Result Image", result_images[i][0])
        if i < len(result_images) / 2:
            cv2.imshow("Warped Source Image", result_images[i][1])
            cv2.imshow("Warped Destination Image", result_images[i][2])
            cv2.imshow("Warped Destination2 Image", image_dest2)
            save_combined_images(i, result_images[i][1], result_images[i][2], result_images[i][0])
        else:
            cv2.imshow("Warped Destination Image", result_images[i][1])
            cv2.imshow("Warped Destination2 Image", result_images[i][2])
            save_combined_images(i, result_images[i][1], result_images[i][2], result_images[i][0])
        cv2.waitKey(DELAY)
cv2.destroyAllWindows()