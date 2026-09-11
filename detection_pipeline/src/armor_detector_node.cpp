/*
 *  READ ALL COMMENTS IN THIS FILE TO UNDERSTAND THE CODE THAT HAS BEEN WRITTEN FOR YOU
 *
 *  This file will create a subscriber node that reads images from the topic where camera_publisher_node publishes
 *  images. It will contain the functionality necessary for locating armor plates within a given image.
 *
 *  Since this file implements a ROS node like camera_publisher_node does, you may want to refer to that file to get
 *  ideas for things you need to consider while writing your code.
 */

#include "../include/armor_detector/armor_detector_node.hpp"

/*
 *  This is the constructor for our subscriber node. It initializes the node inherited from the base class and creates
 *  the subscription to the topic with messages from camera_publisher_node.
 */
ArmorDetectorNode::ArmorDetectorNode() : Node("armor_detector_node"), frame_count(0)
{
    // Subscribe to the camera publisher topic
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));
}

/*
 *  This is an image callback method. It fetches messages (which are images in this case) from the topic this node
 *  subscribes to. The method will also run your armor detection algorithm on the image and show the result.
 *
 *  Callback methods are how we actually read data from a topic. Notice the parameter type and compare it to that of
 *  image_sub_. Our subscription here fetches images, and this method is how you actually operate on that image. Even
 *  though your image processing logic is in a different method, this callback uses those methods as helpers. Make sure
 *  that you understand how callbacks function in a ROS node architecture. If you completed the constructor correctly,
 *  you should know how a callback connects to a subscription in code.
 */
void ArmorDetectorNode::image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // Images are represented by cv::Mat objects. This will be useful when you write image processing logic.
    cv::Mat frame;
    
    // Read the image from the topic into our frame with the proper color space (BGR8)
    try {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    std::vector<cv::RotatedRect> armors = search(frame, lowerHSV, upperHSV, lowerHSV2, upperHSV2);
    frame_count++;

    if (armors.size() == 2)
    {
        auto p0 = rect_to_point(armors[0]);
        auto p1 = rect_to_point(armors[1]);
        std::cout << frame_count << "," << p0[0] << "," << p0[1] << "," << p1[0] << "," << p1[1] << std::endl;

        draw_rotated_rect(frame, armors[0]);
        draw_rotated_rect(frame, armors[1]);
    }
    else
    {
        std::cout << frame_count << "," << "no armor found" << std::endl;
    }

    // Reduce the computational load and just show every 5th image
    if (frame_count % 5 == 0)
    {
        show_frame(frame);
    }
}

/*
 *  This method displays a frame to your screen.
 */
void ArmorDetectorNode::show_frame(cv::Mat &frame)
{
    std::vector<uchar> buf;
    cv::resize(frame, frame, cv::Size(640, 480));
    cv::imencode(".jpg", frame, buf, {cv::IMWRITE_JPEG_QUALITY, 20});
    cv::imshow("Detection Frame", cv::imdecode(buf, cv::IMREAD_COLOR));
    if (cv::waitKey(1) == 27)
    {
        cv::destroyAllWindows();
        rclcpp::shutdown();
    }
}

/*
 *  The main method activates this subscriber node.
 */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}

/*
 *  This method will search a frame for armor plates and return the RotatedRect objects that correspond to the two light
 *  bars that exist on an armor plate.
 */
std::vector<cv::RotatedRect> ArmorDetectorNode::search(cv::Mat& frame, cv::Scalar lowerHSV, cv::Scalar upperHSV, cv::Scalar lowerHSV2, cv::Scalar upperHSV2) {

    // 1) Image Preprocessing
    cv::Mat hsv, blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0); // remove noise
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);        // change color space

    // 2) Color segmentation
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, lowerHSV, upperHSV, mask1);
    cv::inRange(hsv, lowerHSV2, upperHSV2, mask2);
    cv::bitwise_or(mask1, mask2, mask);

    // 2.5) Edge Detection
    // (skipped for now)

    // 3) Contour Detection
    std::vector<std::vector<cv::Point>> contours; // declares a list of object outlines found in an image
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // mask is the input source image, contours is the output container, retrieve external and chain approx

    // 4) Contour Filtering
    std::vector<cv::RotatedRect> light_bars;
    for (const auto &contour : contours) {
        // Minimum area filter — skip contours too small to matter
        if (cv::contourArea(contour) < 10) {
            continue;
        }

        // Fit a rotated rectangle to this contour
        cv::RotatedRect rect = cv::minAreaRect(contour);

        // Check if this shape can be a light bar
        if (is_light_bar(rect)) {
            light_bars.push_back(rect);
        }
    }

    // Check every pair of surviving light bars to find one that forms an armor plate
    for (size_t i = 0; i < light_bars.size(); i++) {
        for (size_t j = i + 1; j < light_bars.size(); j++) {
            cv::RotatedRect &left = light_bars[i].center.x < light_bars[j].center.x ? light_bars[i] : light_bars[j];
            cv::RotatedRect &right = light_bars[i].center.x < light_bars[j].center.x ? light_bars[j] : light_bars[i];

            if (is_armor(left, right)) {
                return {left, right}; // found our one armor plate — stop searching
            }
        }
    }

    return {}; // no valid armor plate found
}

/*
 *  This method draws a rotated rectangle onto a frame. This is used to display the results of your algorithm when you
 *  run the node.
 */
void ArmorDetectorNode::draw_rotated_rect(cv::Mat &frame, cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
    {
        cv::line(frame, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
}

/*
 *  This method determines whether a RotatedRect object can represent a light bar based on the constants defined in the
 *  header file. It checks dimensions, angles, and ratios against our configured thresholds to do so.
 */
bool ArmorDetectorNode::is_light_bar(cv::RotatedRect &rect)
{
    float width = rect.size.width;
    float height = rect.size.height;
    float angle = rect.angle;

    // Verify that the light bar width is valid
    if (width < LIGHT_BAR_WIDTH_LOWER_LIMIT) {
        return false;
    }

    // Verify that the light bar height is valid
    if (height < LIGHT_BAR_HEIGHT_LOWER_LIMIT) {
        return false;
    }

    // Verify that the light bar angle is valid
    // A near-vertical light bar reports as angle near 0 OR near 180, so reject the middle band
    if (angle > LIGHT_BAR_ANGLE_LIMIT && angle < (180.0 - LIGHT_BAR_ANGLE_LIMIT)) {
        return false;
    }

    // Verify that the light bar aspect ratio is valid (height / width)
    float aspect_ratio = height / width;
    if (aspect_ratio < LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT) {
        return false;
    }

    return true;
}

/*
 *  This method determines whether a pair of light bars (RotatedRect objects) can represent an armor plate based on the
 *  constants defined in the header file. It checks dimensions, angles, and ratios against our configured thresholds to
 *  do so.
 */
bool ArmorDetectorNode::is_armor(cv::RotatedRect &left_rect, cv::RotatedRect &right_rect)
{
    // Verify that the light bars are roughly parallel
    float angle_diff = std::abs(left_rect.angle - right_rect.angle);
    float angle_diff_supplement = std::abs(angle_diff - 180.0);
    if (angle_diff > ARMOR_ANGLE_DIFF_LIMIT && angle_diff_supplement > ARMOR_ANGLE_DIFF_LIMIT) {
        return false;
    }

    // Verify that the ratio between the light bar aspect ratios is within the threshold
    float left_ar = left_rect.size.height / left_rect.size.width;
    float right_ar = right_rect.size.height / right_rect.size.width;
    if ((left_ar / right_ar > ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT) ||
        (right_ar / left_ar > ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT)) {
        return false;
    }

    // Verify that the light bars are at roughly the same elevation
    float avg_height = (left_rect.size.height + right_rect.size.height) / 2.0f;
    float y_diff = std::abs(left_rect.center.y - right_rect.center.y) / avg_height;
    if (y_diff > ARMOR_Y_DIFF_LIMIT) {
        return false;
    }

    // Verify that the ratio between light bar heights is within the threshold
    float height_ratio_1 = left_rect.size.height / right_rect.size.height;
    float height_ratio_2 = right_rect.size.height / left_rect.size.height;
    if (height_ratio_1 > ARMOR_HEIGHT_RATIO_LIMIT || height_ratio_2 > ARMOR_HEIGHT_RATIO_LIMIT) {
        return false;
    }

    // Verify that the armor aspect ratio (width / height) is within the threshold
    float plate_width = cv::norm(left_rect.center - right_rect.center);
    float plate_height = avg_height;
    float armor_aspect_ratio = plate_width / plate_height;
    if (armor_aspect_ratio > ARMOR_ASPECT_RATIO_LIMIT) {
        return false;
    }

    return true;
}

/*
 *  This method represents a RotatedRect object as a point and returns it. It exists for debugging output while the node
 *  is being run.
 */
std::vector<cv::Point2f> ArmorDetectorNode::rect_to_point(cv::RotatedRect &rect)
{
    float rad = rect.angle < 90 ? rect.angle * M_PI / 180.f : (rect.angle - 180) * M_PI / 180.f;
    float x_offset = rect.size.height * std::sin(rad) / 2.f;
    float y_offset = rect.size.height * std::cos(rad) / 2.f;

    std::vector<cv::Point2f> points;
    points = std::vector<cv::Point2f>();
    points.push_back(cv::Point2f(int(rect.center.x + x_offset), int(rect.center.y - y_offset)));
    points.push_back(cv::Point2f(int(rect.center.x - x_offset), int(rect.center.y + y_offset)));
    return points;
}
