# THIS WILL SUBSCRIBE TO ZED CAMERA AND PRODUCE AN OCCUPANCY GRID

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
# from bev import CameraProperties
from ultralytics import YOLO
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array
from std_msgs.msg import Header
from math import radians, cos



model = YOLO('drivable_area/drivable_area/utils/nov13.pt')

class CameraProperties(object):
    functional_limit = radians(70.0)
    def __init__(self, height, fov_vert, fov_horz, cameraTilt):
        self.height = float(height)
        self.fov_vert = radians(float(fov_vert))
        self.fov_horz = radians(float(fov_horz))
        self.cameraTilt = radians(float(cameraTilt))
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def src_quad(self, rows, columns):
        if self.bird_src_quad is None:
            self.bird_src_quad = np.array([[0, rows - 1], [columns - 1, rows - 1], [0, 0], [columns - 1, 0]], dtype = 'float32')
        return self.bird_src_quad

    def dst_quad(self, rows, columns, min_angle, max_angle):
        if self.bird_dst_quad is None:
            fov_offset = self.cameraTilt - self.fov_vert/2.0
            bottom_over_top = cos(max_angle + fov_offset)/cos(min_angle + fov_offset)
            bottom_width = columns*bottom_over_top
            blackEdge_width = (columns - bottom_width)/2
            leftX = blackEdge_width
            rightX = leftX + bottom_width
            self.bird_dst_quad = np.array([[leftX, rows], [rightX, rows], [0, 0], [columns, 0]], dtype = 'float32')
        return self.bird_dst_quad

    def reset(self):
        self.bird_src_quad = None
        self.bird_dst_quad = None
        self.matrix = None
        self.maxHeight = None
        self.maxWidth = None
        self.minIndex = None

    def compute_min_index(self, rows, max_angle):
        self.minIndex = int(rows*(1.0 - max_angle/self.fov_vert))
        return self.minIndex

    def compute_max_angle(self):
        return min(CameraProperties.functional_limit - self.cameraTilt + self.fov_vert/2.0, self.fov_vert)

def getBirdView(image, cp):
    rows, columns = image.shape[:2]
    print(rows, columns)
    if columns == 1280:
        columns = 1344
    if rows == 720:
        rows = 752
    min_angle = 0.0
    max_angle = cp.compute_max_angle()
    min_index = cp.compute_min_index(rows, max_angle)
    image = image[min_index:, :]
    rows = image.shape[0]

    src_quad = cp.src_quad(rows, columns)
    dst_quad = cp.dst_quad(rows, columns, min_angle, max_angle)
    warped, bottomLeft, bottomRight, topRight, topLeft = perspective(image, src_quad, dst_quad, cp)
    return warped, bottomLeft, bottomRight, topRight, topLeft, cp.maxWidth, cp.maxHeight

def perspective(image, src_quad, dst_quad, cp):
    bottomLeft, bottomRight, topLeft, topRight = dst_quad
    widthA = topRight[0] - topLeft[0]
    widthB = bottomRight[0] - bottomLeft[0]
    maxWidth1 = max(widthA, widthB)
    heightA = bottomLeft[1] - topLeft[1]
    heightB = bottomRight[1] - topRight[1]
    maxHeight1 = max(heightA, heightB)

    matrix1 = cv2.getPerspectiveTransform(src_quad, dst_quad)
    cp.matrix = matrix1
    cp.maxWidth = int(maxWidth1)
    cp.maxHeight = int(maxHeight1)

    warped = cv2.warpPerspective(image, matrix1, (cp.maxWidth, cp.maxHeight))


    return warped, bottomLeft, bottomRight, topRight, topLeft

class DrivableArea(Node):

    def __init__(self):
        super().__init__('drivable_area')
        self.publisher_ = self.create_publisher(OccupancyGrid, 'topic', 10)
        # the topic 'url' should be changed to a more specific topic name
        self.subscription = self.create_subscription(
            Image,
            'zed_image',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.zed = CameraProperties(54, 68.0, 101.0, 68.0)
        self.curr_pix_size = 0.006
        self.desired_size = 0.05
        self.scale_factor = self.curr_pix_size / self.desired_size

        self.publisher = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)



    def timer_callback(self):
        # Get a new occupancy grid
        grid = self.get_occupancy_grid()
        # Publish the occupancy grid
        self.publisher_.publish(grid)
        # Log a message
        self.get_logger().info('Publishing occupancy grid')
    # TODO: translate the occupancy grid into birds eye view
    # TODO: publish the occupancy grid for nav team
    def listener_callback(self, msg):
        
        #converts Image message to cv2 type
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # runs YOLO predictions on the cv2 image
        results = model.predict(frame, conf=0.25)
        masks = results[0].masks.xy
        #print(masks)
        
        grid = np.zeros((frame.shape[0], frame.shape[1]))
        print(grid.shape)
    
        for mask in masks:
            mask = np.array(mask, dtype=np.int32)
            instance_mask = np.ones((frame.shape[0], frame.shape[1]))
            cv2.fillPoly(instance_mask, [mask], 0)
            np.logical_or(grid, instance_mask)
            
        occupancy_grid_display = grid.astype(np.int8) * 255

        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occupancy_grid_display, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)

        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - 27, bottomRight[1]], [topRight[0] - 65, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        pts = pts.reshape((-1, 1, 2))  # reshape points
        cv2.fillPoly(mask, [pts], True, 0)

        indicies = np.where(mask == -1)
        transformed_image[indicies] = -1

        add_neg = np.full((transformed_image.shape[0], 66), -1, dtype=np.int8)

        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)
        
        transformed_image = np.where(transformed_image==255, 1, transformed_image)
        transformed_image = np.where((transformed_image != 0) & (transformed_image != 1) & (transformed_image != -1), -1, transformed_image)

        new_size = (int(transformed_image.shape[1] * self.scale_factor), int(transformed_image.shape[0] * self.scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)

        rob_arr = np.full((22, 169), -1, dtype=np.int8)
        rob_arr[10][85] = 2

        combined_arr = np.vstack((resized_image, rob_arr))

        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.frame_id = 'map'
        grid.info = MapMetaData()
        grid.info.resolution = self.desired_size
        grid.info.width = combined_arr.shape[1]
        grid.info.height = combined_arr.shape[0]
        grid.info.origin.position.x = float(10)
        grid.info.origin.position.y = float(85)

        grid.data = Array('b', combined_arr.ravel().astype(np.int8))

        self.publisher.publish(grid)

        self.get_logger().info('Publishing occupancy grid')

def main(args=None):
    rclpy.init(args=args)

    node = DrivableArea()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

def numpy_to_occupancy_grid(arr, info=None, x=80,y=77):
    if not len(arr.shape) == 2:
        raise TypeError('Array must be 2D')
    if not arr.dtype == np.int8:
        raise TypeError('Array must be of int8s')

    grid = OccupancyGrid()
    if isinstance(arr, np.ma.MaskedArray):
        # We assume that the masked value are already -1, for speed
        arr = arr.data

    grid.data = Array('b', arr.ravel().astype(np.int8))
    grid.info = info or MapMetaData()
    grid.info.height = arr.shape[0]
    grid.info.width = arr.shape[1]
    grid.info.geometry_msgs.origin = (x,y,0)

#     return grid
if __name__ == '__main__':
    main()