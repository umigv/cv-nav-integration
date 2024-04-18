import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from nav_msgs.msg import OccupancyGrid, MapMetaData
from array import array as Array
from std_msgs.msg import Header
import math
import time
from drivable_area.bev import CameraProperties, getBirdView

# Load the YOLO models for lane and pothole detection
lane_model = YOLO('drivable_area/drivable_area/utils/LLOnly180ep.pt')
hole_model = YOLO('drivable_area/drivable_area/utils/potholesonly100epochs.pt')

UNKNOWN = -1
OCCUPIED = 100
FREE = 0

class DrivableArea(Node):
    def __init__(self):
        super().__init__('drivable_area')
        
        # Get parameters from the parameter server
        self.topic = self.get_parameter('subscription.topic').get_parameter_value().string_value
        self.queue_size = self.get_parameter('subscription.queue_size').get_parameter_value().integer_value
        self.bridge = CvBridge()
        self.zed = CameraProperties(
            self.get_parameter('camera_properties.field_of_view').get_parameter_value().double_value,
            self.get_parameter('camera_properties.aspect_ratio').get_parameter_value().double_value,
            self.get_parameter('camera_properties.near_clip').get_parameter_value().double_value,
            self.get_parameter('camera_properties.far_clip').get_parameter_value().double_value
        )
        self.curr_pix_size = self.get_parameter('pixel_size.current').get_parameter_value().double_value
        self.desired_size = self.get_parameter('pixel_size.desired').get_parameter_value().double_value
        self.scale_factor = self.curr_pix_size / self.desired_size
        self.publisher_topic = self.get_parameter('publisher.topic').get_parameter_value().string_value
        self.publisher_queue_size = self.get_parameter('publisher.queue_size').get_parameter_value().integer_value
        self.lane_model_confidence = self.get_parameter('lane_model.confidence').get_parameter_value().double_value
        self.hole_model_confidence = self.get_parameter('hole_model.confidence').get_parameter_value().double_value

        self.subscription = self.create_subscription(
            Image,
            self.topic,
            self.listener_callback,
            self.queue_size)
        self.subscription  # prevent unused variable warning

        # Create a publisher that publishes OccupancyGrid messages on the 'occupancy_grid' topic
        self.publisher = self.create_publisher(OccupancyGrid, self.publisher_topic, self.publisher_queue_size)

    def get_occupancy_grid(self, frame):
        
        # Predict the lane
        r_lane = lane_model.predict(frame, conf=self.lane_model_confidence)[0]
        image_width, image_height = frame.shape[1], frame.shape[0]
        
        # Create an empty occupancy grid
        occupancy_grid = np.zeros((image_height, image_width))

        # Predict the potholes
        r_hole = hole_model.predict(frame, conf=self.hole_model_confidence)[0]

        # If the lane is detected, fill the occupancy grid with the lane and mark the undrivable area as occupied
    time_of_frame = 0
    if r_lane.masks is not None:
        if(len(r_lane.masks.xy) != 0):
            segment = r_lane.masks.xy[0]
            segment_array = np.array([segment], dtype=np.int32)
            cv2.fillPoly(occupancy_grid, [segment_array], 255)
            time_of_frame = time.time()

    # If the potholes are detected, put a mask of the potholes on the occupancy grid and mark the area as occupied
    if r_hole.boxes is not None:
        for segment in r_hole.boxes.xyxyn:
            x_min, y_min, x_max, y_max = segment
            vertices = np.array([[x_min*self.image_width, y_min*self.image_height], 
                                [x_max*self.image_width, y_min*self.image_height], 
                                [x_max*self.image_width, y_max*self.image_height], 
                                [x_min*self.image_width, y_max*self.image_height]], dtype=np.int32)
            cv2.fillPoly(occupancy_grid, [vertices], color=(0, 0, 0))

    # Calculate the buffer time
    buffer_area = np.sum(occupancy_grid)//255
    buffer_time = math.exp(-buffer_area/(self.image_width*self.image_height)-self.buffer_time_factor)
    return occupancy_grid, buffer_time, time_of_frame
        
    def listener_callback(self, msg):
        
        # Convert the ROS message to a cv2 image
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Resize the image to the dimensions specified in the YAML file
        frame = cv2.resize(frame, (self.image_width, self.image_height))

        # Get the occupancy grid
        occupancy_grid_display, buffer_time, time_of_frame = self.get_occupancy_grid(frame)
        total = np.sum(occupancy_grid_display)
        curr_time = time.time()

        # If the occupancy grid is undetectable, display the previous frame
        if total == 0:
            if curr_time - time_of_frame < self.buffer_time:
                occupancy_grid_display = memory_buffer
            else:
                occupancy_grid_display.fill(255)
        else:
            memory_buffer = occupancy_grid_display

        # Get the bird's eye view of the occupancy grid
        transformed_image, bottomLeft, bottomRight, topRight, topLeft, maxWidth, maxHeight = getBirdView(occupancy_grid_display, self.zed)

        maxHeight = int(maxHeight)
        maxWidth = int(maxWidth)


        # Create a mask to remove the area outside the drivable area
        mask = np.full((maxHeight, maxWidth), -1, dtype=np.int8)
        pts =  np.array([bottomLeft, [bottomRight[0] - self.right_bottom_offset, bottomRight[1]], [topRight[0] - self.right_top_offset, topRight[1]], topLeft])
        pts = pts.astype(np.int32)  # convert points to int32
        pts = pts.reshape((-1, 1, 2))  # reshape points
        cv2.fillPoly(mask, [pts], True, 0)

        # Apply the mask to the occupancy grid
        indicies = np.where(mask == -1)
        transformed_image[indicies] = -1

        # Add a negative border to the occupancy grid
        add_neg = np.full((transformed_image.shape[0], self.neg_border_width), -1, dtype=np.int8)

        # Concatenate the negative border to the occupancy grid
        transformed_image = np.concatenate((add_neg, transformed_image), axis=1)

        # Convert the occupancy grid to a binary grid
        transformed_image = np.where(transformed_image==255, 1, transformed_image)
        transformed_image = np.where((transformed_image != 0) & (transformed_image != 1) & (transformed_image != -1), -1, transformed_image)

        new_size = (int(transformed_image.shape[1] * self.scale_factor), int(transformed_image.shape[0] * self.scale_factor))
        resized_image = cv2.resize(transformed_image, new_size, interpolation = cv2.INTER_NEAREST_EXACT)

        # Create a robot occupancy grid to display the robot's position
        rob_arr = np.full((self.robot_grid_height, self.robot_grid_width), -1, dtype=np.int8)
        rob_arr[self.robot_position_y][self.robot_position_x] = 2

        # Concatenate the robot occupancy grid to the occupancy grid
        combined_arr = np.vstack((resized_image, rob_arr))

        combined_arr = np.where(combined_arr==self.empty_value, self.temp_value, combined_arr)
        combined_arr = np.where(combined_arr==self.occupied_value, self.empty_value, combined_arr)
        combined_arr = np.where(combined_arr==self.temp_value, self.occupied_value, combined_arr)

        # np.savetxt('occupancy_grid.txt', combined_arr, fmt='%d')
                
        self.send_occupancy_grid(combined_arr)

        def send_occupancy_grid(self, array):
            grid = OccupancyGrid()
            grid.header = Header()
            grid.header.stamp = self.get_clock().now().to_msg()
            grid.header.frame_id = self.frame_id
            grid.info = MapMetaData()
            grid.info.resolution = self.desired_size
            grid.info.width = array.shape[1]
            grid.info.height = array.shape[0]
            grid.info.origin.position.x = self.origin_x
            grid.info.origin.position.y = self.origin_y
            grid.info.origin.position.z = self.origin_z

            grid.data = Array('b', array.ravel().astype(np.int8))

            self.publisher.publish(grid)
            self.get_logger().info('Publishing occupancy grid')


def main(args=None):
    rclpy.init(args=args)

    # Get the path of the YAML file from the command line arguments
    yaml_file_path = args[1] if len(args) > 1 else 'default/path/to/your/yaml/file.yaml'

    node = DrivableArea(yaml_file_path)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main(sys.argv)