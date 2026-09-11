/*
 *  READ ALL COMMENTS IN THIS FILE TO UNDERSTAND THE CODE THAT HAS BEEN WRITTEN FOR YOU
 *
 *  This file will create a subscriber node that reads images from the topic where camera_publisher_node publishes
 *  images. It will contain the functionality necessary for locating armor plates within a given image.
 */

#include "../include/armor_detector/armor_detector_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

/*
 *  Constructor
 */
ArmorDetectorNode::ArmorDetectorNode()
    : Node("armor_detector_node"), frame_count(0)
{
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "camera/image_raw",
        rclcpp::SensorDataQoS(),
        std::bind(
            &ArmorDetectorNode::image_callback,
            this,
            std::placeholders::_1));
}

/*
 *  Process each incoming image.
 */
void ArmorDetectorNode::image_callback(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    cv::Mat frame;

    try
    {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }
    catch (const cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(
            this->get_logger(),
            "cv_bridge exception: %s",
            e.what());
        return;
    }

    if (frame.empty())
    {
        RCLCPP_WARN(this->get_logger(), "Received empty frame");
        return;
    }

    frame_count++;

    std::vector<cv::RotatedRect> armors =
        search(frame, lowerHSV, upperHSV, lowerHSV2, upperHSV2);

    if (armors.size() == 2)
    {
        auto p0 = rect_to_point(armors[0]);
        auto p1 = rect_to_point(armors[1]);

        std::cout
            << frame_count << ","
            << p0[0] << ","
            << p0[1] << ","
            << p1[0] << ","
            << p1[1]
            << std::endl;

        draw_rotated_rect(frame, armors[0]);
        draw_rotated_rect(frame, armors[1]);

        // Draw the center of the detected armor.
        cv::Point center(
            static_cast<int>(
                (armors[0].center.x + armors[1].center.x) / 2.0f),
            static_cast<int>(
                (armors[0].center.y + armors[1].center.y) / 2.0f));

        cv::circle(
            frame,
            center,
            4,
            cv::Scalar(255, 0, 0),
            -1);
    }
    else
    {
        std::cout
            << frame_count
            << ",no armor found"
            << std::endl;
    }

    // Only display every fifth frame to reduce computational load.
    if (frame_count % 5 == 0)
    {
        show_frame(frame);
    }
}

/*
 *  Display a frame.
 */
void ArmorDetectorNode::show_frame(cv::Mat &frame)
{
    if (frame.empty())
    {
        return;
    }

    cv::Mat display_frame;

    // Resize a copy rather than modifying the original frame.
    cv::resize(
        frame,
        display_frame,
        cv::Size(640, 480));

    cv::imshow(
        "Detection Frame",
        display_frame);

    if (cv::waitKey(1) == 27)
    {
        cv::destroyAllWindows();
        rclcpp::shutdown();
    }
}

/*
 *  Main method.
 */
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    rclcpp::spin(
        std::make_shared<ArmorDetectorNode>());

    rclcpp::shutdown();

    return 0;
}

/*
 *  Search the image for armor plates.
 *
 *  The process is:
 *
 *  1. Blur image
 *  2. Convert BGR -> HSV
 *  3. Segment the target color
 *  4. Clean the binary mask
 *  5. Find contours
 *  6. Convert contours into rotated rectangles
 *  7. Filter possible light bars
 *  8. Test every pair of light bars
 *  9. Return the best valid pair
 */
std::vector<cv::RotatedRect> ArmorDetectorNode::search(
    cv::Mat &frame,
    cv::Scalar lowerHSV,
    cv::Scalar upperHSV,
    cv::Scalar lowerHSV2,
    cv::Scalar upperHSV2)
{
    cv::Mat hsv;
    cv::Mat blurred;

    /*
     * 1) Image preprocessing
     */
    cv::GaussianBlur(
        frame,
        blurred,
        cv::Size(5, 5),
        0);

    cv::cvtColor(
        blurred,
        hsv,
        cv::COLOR_BGR2HSV);

    /*
     * 2) Color segmentation
     *
     * Red wraps around the HSV hue range, so we need
     * two masks.
     */
    cv::Mat mask1;
    cv::Mat mask2;
    cv::Mat mask;

    cv::inRange(
        hsv,
        lowerHSV,
        upperHSV,
        mask1);

    cv::inRange(
        hsv,
        lowerHSV2,
        upperHSV2,
        mask2);

    cv::bitwise_or(
        mask1,
        mask2,
        mask);

    /*
     * 3) Clean up the binary mask.
     *
     * Opening removes small isolated noise.
     * Closing fills small gaps inside light bars.
     */
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(3, 3));

    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_OPEN,
        kernel);

    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        kernel);

    /*
     * 4) Find contours.
     */
    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(
        mask,
        contours,
        cv::RETR_EXTERNAL,
        cv::CHAIN_APPROX_SIMPLE);

    /*
     * 5) Convert contours into candidate light bars.
     */
    std::vector<cv::RotatedRect> light_bars;

    for (const auto &contour : contours)
    {
        double area = cv::contourArea(contour);

        // Ignore tiny regions.
        if (area < 10.0)
        {
            continue;
        }

        cv::RotatedRect rect =
            cv::minAreaRect(contour);

        if (is_light_bar(rect))
        {
            light_bars.push_back(rect);
        }
    }

    /*
     * 6) Find the best pair of light bars.
     *
     * Instead of immediately returning the first valid pair,
     * evaluate all pairs and choose the one with the smallest
     * geometric error.
     */
    float best_score =
        std::numeric_limits<float>::max();

    cv::RotatedRect best_left;
    cv::RotatedRect best_right;

    bool found_armor = false;

    for (size_t i = 0; i < light_bars.size(); i++)
    {
        for (size_t j = i + 1; j < light_bars.size(); j++)
        {
            cv::RotatedRect left;
            cv::RotatedRect right;

            if (light_bars[i].center.x <
                light_bars[j].center.x)
            {
                left = light_bars[i];
                right = light_bars[j];
            }
            else
            {
                left = light_bars[j];
                right = light_bars[i];
            }

            if (!is_armor(left, right))
            {
                continue;
            }

            /*
             * Calculate a score for this pair.
             *
             * Good armor should have:
             * - similar heights
             * - similar angles
             * - similar aspect ratios
             * - similar vertical position
             */
            float left_width =
                std::min(
                    left.size.width,
                    left.size.height);

            float left_height =
                std::max(
                    left.size.width,
                    left.size.height);

            float right_width =
                std::min(
                    right.size.width,
                    right.size.height);

            float right_height =
                std::max(
                    right.size.width,
                    right.size.height);

            float avg_height =
                (left_height + right_height) / 2.0f;

            if (avg_height <= 0.0f)
            {
                continue;
            }

            float height_difference =
                std::abs(left_height - right_height)
                / avg_height;

            float y_difference =
                std::abs(
                    left.center.y -
                    right.center.y)
                / avg_height;

            float left_aspect =
                left_height / std::max(left_width, 0.001f);

            float right_aspect =
                right_height / std::max(right_width, 0.001f);

            float aspect_difference =
                std::abs(
                    left_aspect -
                    right_aspect);

            float score =
                height_difference
                + y_difference
                + 0.25f * aspect_difference;

            if (score < best_score)
            {
                best_score = score;

                best_left = left;
                best_right = right;

                found_armor = true;
            }
        }
    }

    if (found_armor)
    {
        return {
            best_left,
            best_right
        };
    }

    return {};
}

/*
 *  Draw a rotated rectangle.
 */
void ArmorDetectorNode::draw_rotated_rect(
    cv::Mat &frame,
    cv::RotatedRect &rect)
{
    cv::Point2f vertices[4];

    rect.points(vertices);

    for (int i = 0; i < 4; i++)
    {
        cv::line(
            frame,
            vertices[i],
            vertices[(i + 1) % 4],
            cv::Scalar(0, 255, 0),
            2);
    }
}

/*
 *  Determine whether a rectangle could represent
 *  a light bar.
 */
bool ArmorDetectorNode::is_light_bar(
    cv::RotatedRect &rect)
{
    /*
     * minAreaRect() does not guarantee that width is
     * the short side and height is the long side.
     *
     * Therefore explicitly determine them.
     */
    float width =
        std::min(
            rect.size.width,
            rect.size.height);

    float height =
        std::max(
            rect.size.width,
            rect.size.height);

    /*
     * Reject extremely small objects.
     */
    if (width < LIGHT_BAR_WIDTH_LOWER_LIMIT)
    {
        return false;
    }

    if (height < LIGHT_BAR_HEIGHT_LOWER_LIMIT)
    {
        return false;
    }

    /*
     * Light bars should be substantially taller
     * than they are wide.
     */
    float aspect_ratio =
        height / width;

    if (aspect_ratio <
        LIGHT_BAR_ASPECT_RATIO_LOWER_LIMIT)
    {
        return false;
    }

    /*
     * Determine the orientation of the long side
     * using the four rectangle vertices.
     *
     * This avoids relying directly on OpenCV's
     * RotatedRect angle convention.
     */
    cv::Point2f points[4];
    rect.points(points);

    float longest_length = 0.0f;
    cv::Point2f longest_vector;

    for (int i = 0; i < 4; i++)
    {
        cv::Point2f vector =
            points[(i + 1) % 4] - points[i];

        float length =
            cv::norm(vector);

        if (length > longest_length)
        {
            longest_length = length;
            longest_vector = vector;
        }
    }

    /*
     * Calculate angle of the long side relative
     * to the horizontal.
     */
    float angle =
        std::atan2(
            longest_vector.y,
            longest_vector.x)
        * 180.0f
        / static_cast<float>(M_PI);

    angle = std::abs(angle);

    /*
     * Convert the angle to [0, 90].
     */
    if (angle > 90.0f)
    {
        angle = 180.0f - angle;
    }

    /*
     * A light bar should be approximately vertical.
     */
    if (std::abs(90.0f - angle) >
        LIGHT_BAR_ANGLE_LIMIT)
    {
        return false;
    }

    return true;
}

/*
 *  Determine whether two light bars form an armor plate.
 */
bool ArmorDetectorNode::is_armor(
    cv::RotatedRect &left_rect,
    cv::RotatedRect &right_rect)
{
    /*
     * Normalize the dimensions of each rectangle.
     */
    float left_width =
        std::min(
            left_rect.size.width,
            left_rect.size.height);

    float left_height =
        std::max(
            left_rect.size.width,
            left_rect.size.height);

    float right_width =
        std::min(
            right_rect.size.width,
            right_rect.size.height);

    float right_height =
        std::max(
            right_rect.size.width,
            right_rect.size.height);

    /*
     * Prevent division by zero.
     */
    if (left_width <= 0.0f ||
        right_width <= 0.0f ||
        left_height <= 0.0f ||
        right_height <= 0.0f)
    {
        return false;
    }

    /*
     * Calculate aspect ratios.
     */
    float left_ar =
        left_height / left_width;

    float right_ar =
        right_height / right_width;

    /*
     * Light bars should have similar aspect ratios.
     */
    float aspect_ratio =
        std::max(left_ar, right_ar) /
        std::min(left_ar, right_ar);

    if (aspect_ratio >
        ARMOR_LIGHT_BAR_ASPECT_RATIO_RATIO_LIMIT)
    {
        return false;
    }

    /*
     * Light bars should have similar heights.
     */
    float height_ratio =
        std::max(left_height, right_height) /
        std::min(left_height, right_height);

    if (height_ratio >
        ARMOR_HEIGHT_RATIO_LIMIT)
    {
        return false;
    }

    /*
     * Light bars should be at approximately the same
     * vertical position.
     */
    float avg_height =
        (left_height + right_height) / 2.0f;

    float y_difference =
        std::abs(
            left_rect.center.y -
            right_rect.center.y)
        / avg_height;

    if (y_difference >
        ARMOR_Y_DIFF_LIMIT)
    {
        return false;
    }

    /*
     * Calculate the orientation of each light bar
     * using the longest side of its bounding box.
     */
    auto get_long_side_angle =
        [](const cv::RotatedRect &rect)
    {
        cv::Point2f points[4];
        rect.points(points);

        float longest_length = 0.0f;
        cv::Point2f longest_vector;

        for (int i = 0; i < 4; i++)
        {
            cv::Point2f vector =
                points[(i + 1) % 4] - points[i];

            float length =
                cv::norm(vector);

            if (length > longest_length)
            {
                longest_length = length;
                longest_vector = vector;
            }
        }

        float angle =
            std::atan2(
                longest_vector.y,
                longest_vector.x)
            * 180.0f
            / static_cast<float>(M_PI);

        return angle;
    };

    float left_angle =
        get_long_side_angle(left_rect);

    float right_angle =
        get_long_side_angle(right_rect);

    /*
     * Find the smallest difference between the
     * two orientations.
     */
    float angle_difference =
        std::abs(left_angle - right_angle);

    if (angle_difference > 180.0f)
    {
        angle_difference =
            360.0f - angle_difference;
    }

    if (angle_difference > 90.0f)
    {
        angle_difference =
            180.0f - angle_difference;
    }

    if (angle_difference >
        ARMOR_ANGLE_DIFF_LIMIT)
    {
        return false;
    }

    /*
     * Distance between the two light bars.
     */
    float plate_width =
        cv::norm(
            left_rect.center -
            right_rect.center);

    /*
     * An armor plate should not be extremely wide
     * compared with the height of its light bars.
     */
    float armor_aspect_ratio =
        plate_width / avg_height;

    if (armor_aspect_ratio >
        ARMOR_ASPECT_RATIO_LIMIT)
    {
        return false;
    }

    /*
     * Also make sure the two bars aren't almost
     * on top of one another.
     */
    if (plate_width < avg_height * 0.5f)
    {
        return false;
    }

    return true;
}

/*
 *  Return two points along the long axis of the
 *  light bar.
 *
 *  These points are primarily used for debugging output.
 */
std::vector<cv::Point2f> ArmorDetectorNode::rect_to_point(
    cv::RotatedRect &rect)
{
    cv::Point2f points[4];
    rect.points(points);

    /*
     * Find the longest edge.
     */
    float longest_length = 0.0f;
    cv::Point2f longest_vector;

    for (int i = 0; i < 4; i++)
    {
        cv::Point2f vector =
            points[(i + 1) % 4] - points[i];

        float length =
            cv::norm(vector);

        if (length > longest_length)
        {
            longest_length = length;
            longest_vector = vector;
        }
    }

    /*
     * Normalize the vector.
     */
    float length =
        cv::norm(longest_vector);

    if (length <= 0.0f)
    {
        return {
            rect.center,
            rect.center
        };
    }

    longest_vector /= length;

    /*
     * Use half the long-side length to find
     * the two endpoints.
     */
    float half_height =
        std::max(
            rect.size.width,
            rect.size.height) / 2.0f;

    cv::Point2f point1 =
        rect.center +
        longest_vector * half_height;

    cv::Point2f point2 =
        rect.center -
        longest_vector * half_height;

    return {
        point1,
        point2
    };
}