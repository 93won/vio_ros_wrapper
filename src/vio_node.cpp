#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <deque>
#include <thread>
#include <atomic>

// lightweight_vio includes
#include "processing/Estimator.h"
#include "ThreadSafeQueue.h"
#include "database/Frame.h"
#include "database/Feature.h"
#include "database/MapPoint.h"
#include "util/Config.h"

using namespace std::placeholders;
using namespace lightweight_vio;

// IMU measurement data
struct ImuData {
    double timestamp;
    double acc_x, acc_y, acc_z;      // Linear acceleration (m/s^2)
    double gyro_x, gyro_y, gyro_z;   // Angular velocity (rad/s)
    
    ImuData(double t, double ax, double ay, double az, 
            double gx, double gy, double gz)
        : timestamp(t), acc_x(ax), acc_y(ay), acc_z(az),
          gyro_x(gx), gyro_y(gy), gyro_z(gz) {}
    
    // Convert to lightweight_vio::IMUData
    IMUData to_vio_imu_data() const {
        return IMUData(timestamp, 
                      Eigen::Vector3f(acc_x, acc_y, acc_z),
                      Eigen::Vector3f(gyro_x, gyro_y, gyro_z));
    }
};

// VIO measurement dataset: stereo images OR rgbd + IMU bundle
struct VioDataset {
    // Stereo images (for stereo mode)
    cv::Mat left_image;
    cv::Mat right_image;
    
    // RGBD images (for rgbd mode)
    cv::Mat rgb_image;
    cv::Mat depth_image;
    
    double image_timestamp;
    bool is_rgbd;  // true for RGBD, false for stereo
    
    // IMU measurements between previous and current image
    std::vector<ImuData> imu_bundle;
    
    // Metadata
    size_t dataset_id;              // Sequential ID
    size_t imu_count;               // Number of IMU measurements
    double time_span;               // Time duration of IMU bundle
    
    VioDataset() 
        : image_timestamp(0.0), is_rgbd(false), dataset_id(0), imu_count(0), time_span(0.0) {}
    
    // Get statistics
    double getAverageImuRate() const {
        return (time_span > 0 && imu_count > 0) ? imu_count / time_span : 0.0;
    }
    
    bool isValid() const {
        if (is_rgbd) {
            return !rgb_image.empty() && !depth_image.empty();
        } else {
            return !left_image.empty() && !right_image.empty();
        }
    }
    
    std::string toString() const {
        char buffer[512];
        const char* mode_str = is_rgbd ? "RGBD" : "Stereo";
        int img_width = is_rgbd ? rgb_image.cols : left_image.cols;
        int img_height = is_rgbd ? rgb_image.rows : left_image.rows;
        
        if (imu_count > 0) {
            double imu_start_time = imu_bundle.front().timestamp;
            double imu_end_time = imu_bundle.back().timestamp;
            snprintf(buffer, sizeof(buffer),
                    "Dataset #%zu [%s] | Image: t=%.6f | IMU: count=%zu, start=%.6f, end=%.6f, span=%.4fs (%.1f Hz) | Size=[%dx%d]",
                    dataset_id, mode_str, image_timestamp, imu_count, 
                    imu_start_time, imu_end_time, time_span, getAverageImuRate(),
                    img_width, img_height);
        } else {
            snprintf(buffer, sizeof(buffer),
                    "Dataset #%zu [%s] | Image: t=%.6f | IMU: count=0 (NO IMU DATA) | Size=[%dx%d]",
                    dataset_id, mode_str, image_timestamp, 
                    img_width, img_height);
        }
        return std::string(buffer);
    }
};

// ⭐ VIO processing result for publisher thread
struct VIOProcessingResult {
    bool success;
    Eigen::Matrix4f Twb;  // Body to world transform
    double timestamp;
    int num_features;
    int num_inliers;
    std::shared_ptr<Frame> current_frame;
    
    VIOProcessingResult() : success(false), num_features(0), num_inliers(0) {
        Twb = Eigen::Matrix4f::Identity();
    }
};

class VIONode : public rclcpp::Node
{
public:
    VIONode() : Node("vio_node"), 
                stereo_count_(0), 
                rgbd_count_(0),
                imu_count_(0),
                left_count_(0),
                right_count_(0),
                imu_pivot_index_(0),
                last_image_time_(-1.0),
                dataset_id_(0),
                is_rgbd_mode_(false),
                running_(true)  // ⭐ Thread running flag
    {
        // Declare parameters
        this->declare_parameter<std::string>("left_image_topic", "/cam0/image_raw");
        this->declare_parameter<std::string>("right_image_topic", "/cam1/image_raw");
        this->declare_parameter<std::string>("rgb_image_topic", "/camera/color/image_raw");
        this->declare_parameter<std::string>("depth_image_topic", "/camera/depth/image_rect_raw");
        this->declare_parameter<std::string>("imu_topic", "/imu0");
        this->declare_parameter<int>("queue_size", 10);
        this->declare_parameter<double>("imu_time_tolerance", 0.005); // 5ms
        this->declare_parameter<std::string>("config_file", "");

        // Get parameters
        std::string left_topic = this->get_parameter("left_image_topic").as_string();
        std::string right_topic = this->get_parameter("right_image_topic").as_string();
        std::string rgb_topic = this->get_parameter("rgb_image_topic").as_string();
        std::string depth_topic = this->get_parameter("depth_image_topic").as_string();
        std::string imu_topic = this->get_parameter("imu_topic").as_string();
        int queue_size = this->get_parameter("queue_size").as_int();
        imu_time_tolerance_ = this->get_parameter("imu_time_tolerance").as_double();
        std::string config_path = this->get_parameter("config_file").as_string();

        RCLCPP_INFO(this->get_logger(), "Starting VIO Node");
        RCLCPP_INFO(this->get_logger(), "IMU topic: %s", imu_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "IMU time tolerance: %.1f ms", imu_time_tolerance_ * 1000);
        RCLCPP_INFO(this->get_logger(), "Config file: %s", config_path.c_str());

        // Initialize lightweight_vio estimator
        try {
            if (config_path.empty()) {
                throw std::runtime_error("Config file path not specified! Please set 'config_file' parameter.");
            }
            
            if (!Config::getInstance().load(config_path)) {
                throw std::runtime_error("Failed to load config file: " + config_path);
            }
            
            // ⭐ Detect camera type from config
            is_rgbd_mode_ = Config::getInstance().is_rgbd();
            
            if (is_rgbd_mode_) {
                RCLCPP_INFO(this->get_logger(), "🎥 RGBD Mode Enabled");
                RCLCPP_INFO(this->get_logger(), "RGB image topic: %s", rgb_topic.c_str());
                RCLCPP_INFO(this->get_logger(), "Depth image topic: %s", depth_topic.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "🎥 Stereo Mode Enabled");
                RCLCPP_INFO(this->get_logger(), "Left image topic: %s", left_topic.c_str());
                RCLCPP_INFO(this->get_logger(), "Right image topic: %s", right_topic.c_str());
            }
            
            estimator_ = std::make_shared<Estimator>();
            RCLCPP_INFO(this->get_logger(), "VIO Estimator initialized");
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize VIO estimator: %s", e.what());
            throw;
        }

        // Create subscribers based on camera type
        if (is_rgbd_mode_) {
            // RGBD mode: subscribe to RGB + Depth
            rgb_image_sub_.subscribe(this, rgb_topic);
            depth_image_sub_.subscribe(this, depth_topic);
            
            // Synchronize RGB + Depth images
            rgbd_sync_ = std::make_shared<message_filters::Synchronizer<
                message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::Image, 
                    sensor_msgs::msg::Image>>>(
                message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::Image, 
                    sensor_msgs::msg::Image>(queue_size), 
                rgb_image_sub_, depth_image_sub_);
            
            rgbd_sync_->registerCallback(&VIONode::rgbdImageCallback, this);
        } else {
            // Stereo mode: subscribe to Left + Right
            left_image_sub_.subscribe(this, left_topic);
            right_image_sub_.subscribe(this, right_topic);

            // Synchronize stereo images
            sync_ = std::make_shared<message_filters::Synchronizer<
                message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::Image, 
                    sensor_msgs::msg::Image>>>(
                message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::Image, 
                    sensor_msgs::msg::Image>(queue_size), 
                left_image_sub_, right_image_sub_);
            
            sync_->registerCallback(&VIONode::stereoImageCallback, this);
        }

        // IMU subscriber
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, queue_size,
            [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {
                this->imuCallback(msg);
            });

        // Odometry publisher
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/vio/odometry", 10);
        
        // Pose publisher
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
            "/vio/pose", 10);
        
        // Map point cloud publishers
        current_frame_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/vio/current_frame_points", 10);
        
        keyframe_points_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/vio/keyframe_points", 10);
        
        // Trajectory publisher
        trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(
            "/vio/trajectory", 10);
        
        // Camera frustum publisher
        camera_frustum_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "/vio/camera_frustum", 10);
        
        // Tracking image publisher
        tracking_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/vio/tracking_image", 10);
        
        // Dense point cloud publisher (RGBD only)
        dense_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/vio/dense_cloud", 10);
        
        // Depth image publisher (RGBD only)
        depth_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/vio/depth_image", 10);

        // TF broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // ⭐ Start worker threads (Producer-Consumer pattern)
        processing_thread_ = std::thread(&VIONode::processingThreadLoop, this);
        publisher_thread_ = std::thread(&VIONode::publisherThreadLoop, this);

        RCLCPP_INFO(this->get_logger(), "VIO Node initialized successfully");
        RCLCPP_INFO(this->get_logger(), "✅ Processing & Publisher threads started");
    }
    
    // ⭐ Destructor: shutdown threads gracefully
    ~VIONode() {
        RCLCPP_INFO(this->get_logger(), "Shutting down VIO Node...");
        
        // Stop threads
        running_ = false;
        image_queue_.shutdown();
        result_queue_.shutdown();
        
        // Wait for threads to finish
        if (processing_thread_.joinable()) {
            processing_thread_.join();
            RCLCPP_INFO(this->get_logger(), "Processing thread stopped");
        }
        
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
            RCLCPP_INFO(this->get_logger(), "Publisher thread stopped");
        }
        
        RCLCPP_INFO(this->get_logger(), "VIO Node shutdown complete");
    }

private:
    // ⭐ LIGHTWEIGHT Callback: 데이터만 queue에 넣고 즉시 return (Stereo)
    void stereoImageCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& right_msg)
    {
        try {
            stereo_count_++;
            
            // Convert ROS images to OpenCV format (빠름)
            cv_bridge::CvImageConstPtr left_ptr = cv_bridge::toCvShare(left_msg, "mono8");
            cv_bridge::CvImageConstPtr right_ptr = cv_bridge::toCvShare(right_msg, "mono8");

            cv::Mat left_image = left_ptr->image.clone();  // Clone for thread safety
            cv::Mat right_image = right_ptr->image.clone();

            double current_image_time = left_msg->header.stamp.sec + 
                                       left_msg->header.stamp.nanosec * 1e-9;

            // Store first and last timestamps
            if (stereo_count_ == 1) {
                first_stereo_time_ = current_image_time;
                last_image_time_ = current_image_time;
                RCLCPP_INFO(this->get_logger(), 
                           "First stereo image received at %.6f - initializing VIO",
                           current_image_time);
            }
            last_stereo_time_ = current_image_time;

            // Build VIO dataset (빠름 - IMU data copy만)
            VioDataset dataset = buildVioDataset(left_image, right_image, 
                                                 current_image_time);
            
            // ⭐ Queue에 push만 하고 바로 return (NO BLOCKING!)
            image_queue_.push(dataset);
            
            size_t queue_size = image_queue_.size();
            if (queue_size > 10) {  // Warn if queue is growing
                RCLCPP_WARN(this->get_logger(), 
                           "⚠️  Image queue size: %zu (processing may be slower than incoming data)",
                           queue_size);
            }
            
            last_image_time_ = current_image_time;

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    // ⭐ RGBD Callback: RGB + Depth 동기화
    void rgbdImageCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& rgb_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& depth_msg)
    {
        try {
            rgbd_count_++;
            
            // Convert ROS images to OpenCV format
            cv_bridge::CvImageConstPtr rgb_ptr = cv_bridge::toCvShare(rgb_msg, "bgr8");
            cv_bridge::CvImageConstPtr depth_ptr = cv_bridge::toCvShare(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);

            cv::Mat rgb_image = rgb_ptr->image.clone();  // Clone for thread safety
            cv::Mat depth_image = depth_ptr->image.clone();
            
            // Convert depth from uint16 (mm) to float32 (m)
            cv::Mat depth_float;
            depth_image.convertTo(depth_float, CV_32FC1, 0.001);  // mm to m

            double current_image_time = rgb_msg->header.stamp.sec + 
                                       rgb_msg->header.stamp.nanosec * 1e-9;

            // Store first and last timestamps
            if (rgbd_count_ == 1) {
                first_rgbd_time_ = current_image_time;
                last_image_time_ = current_image_time;
                RCLCPP_INFO(this->get_logger(), 
                           "First RGBD image received at %.6f - initializing VIO",
                           current_image_time);
            }
            last_rgbd_time_ = current_image_time;

            // Build RGBD dataset
            VioDataset dataset = buildRGBDDataset(rgb_image, depth_float, current_image_time);
            
            // ⭐ Queue에 push만 하고 바로 return (NO BLOCKING!)
            image_queue_.push(dataset);
            
            size_t queue_size = image_queue_.size();
            if (queue_size > 10) {
                RCLCPP_WARN(this->get_logger(), 
                           "⚠️  Image queue size: %zu (processing may be slower than incoming data)",
                           queue_size);
            }
            
            last_image_time_ = current_image_time;

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& msg)
    {
        imu_count_++;
        
        double timestamp = msg->header.stamp.sec + 
                         msg->header.stamp.nanosec * 1e-9;

        // Store first and last timestamps
        if (imu_count_ == 1) {
            first_imu_time_ = timestamp;
        }
        last_imu_time_ = timestamp;

        // Extract IMU data and add to buffer
        ImuData imu_data(
            timestamp,
            msg->linear_acceleration.x,
            msg->linear_acceleration.y,
            msg->linear_acceleration.z,
            msg->angular_velocity.x,
            msg->angular_velocity.y,
            msg->angular_velocity.z
        );
        
        imu_buffer_.push_back(imu_data);

        RCLCPP_DEBUG(this->get_logger(),
                   "IMU #%ld at %.6f - Buffer size: %zu",
                   imu_count_, timestamp, imu_buffer_.size());
    }

    VioDataset buildVioDataset(const cv::Mat& left_image, 
                               const cv::Mat& right_image,
                               double current_image_time)
    {
        VioDataset dataset;
        dataset.dataset_id = ++dataset_id_;
        dataset.is_rgbd = false;
        dataset.left_image = left_image.clone();
        dataset.right_image = right_image.clone();
        dataset.image_timestamp = current_image_time;
        
        // Extract IMU measurements
        extractIMUData(dataset, current_image_time);
        
        return dataset;
    }
    
    VioDataset buildRGBDDataset(const cv::Mat& rgb_image,
                                const cv::Mat& depth_image,
                                double current_image_time)
    {
        VioDataset dataset;
        dataset.dataset_id = ++dataset_id_;
        dataset.is_rgbd = true;
        dataset.rgb_image = rgb_image.clone();
        dataset.depth_image = depth_image.clone();
        dataset.image_timestamp = current_image_time;
        
        // Extract IMU measurements
        extractIMUData(dataset, current_image_time);
        
        return dataset;
    }
    
    void extractIMUData(VioDataset& dataset, double current_image_time)
    {
        // Extract IMU measurements between last_image_time and current_image_time
        size_t start_idx = imu_pivot_index_;
        
        for (size_t i = start_idx; i < imu_buffer_.size(); i++) {
            double imu_time = imu_buffer_[i].timestamp;
            
            // Skip IMU data before last image time (with tolerance)
            if (imu_time < last_image_time_ - imu_time_tolerance_) {
                continue;
            }
            
            // Include IMU data up to current image time + tolerance
            if (imu_time <= current_image_time + imu_time_tolerance_) {
                dataset.imu_bundle.push_back(imu_buffer_[i]);
                imu_pivot_index_ = i + 1; // Move pivot forward
            } else {
                // IMU data is beyond current image time
                break;
            }
        }
        
        // Calculate metadata
        dataset.imu_count = dataset.imu_bundle.size();
        if (dataset.imu_count > 0) {
            dataset.time_span = dataset.imu_bundle.back().timestamp - 
                               dataset.imu_bundle.front().timestamp;
        }
    }
    
    // ⭐ Processing Thread: VIO 처리 (무거운 작업, 여기서 blocking 해도 callback 안 막힘!)
    void processingThreadLoop() {
        RCLCPP_INFO(this->get_logger(), "🔧 Processing thread started");
        
        size_t processed_count = 0;
        
        while (running_) {
            // Queue에서 dataset pop (blocking)
            auto dataset_opt = image_queue_.pop();
            
            if (!dataset_opt.has_value()) {
                // Queue shutdown signal
                break;
            }
            
            VioDataset dataset = dataset_opt.value();
            processed_count++;
            
            // Process VIO dataset
            VIOProcessingResult result;
            result.timestamp = dataset.image_timestamp;
            
            if (!estimator_) {
                RCLCPP_ERROR(this->get_logger(), "Estimator not initialized!");
                continue;
            }

            try {
                // Convert timestamp to nanoseconds
                long long timestamp_ns = static_cast<long long>(dataset.image_timestamp * 1e9);
                
                // Convert IMU data
                std::vector<IMUData> vio_imu_data;
                vio_imu_data.reserve(dataset.imu_bundle.size());
                for (const auto& imu : dataset.imu_bundle) {
                    vio_imu_data.push_back(imu.to_vio_imu_data());
                }
                
                // Process frame based on mode
                Estimator::EstimationResult est_result;
                
                if (dataset.is_rgbd) {
                    // ⭐ RGBD mode
                    // Preprocess RGB image (histogram equalization)
                    cv::Mat processed_rgb;
                    if (dataset.rgb_image.channels() == 3) {
                        cv::Mat gray;
                        cv::cvtColor(dataset.rgb_image, gray, cv::COLOR_BGR2GRAY);
                        cv::equalizeHist(gray, processed_rgb);
                    } else {
                        cv::equalizeHist(dataset.rgb_image, processed_rgb);
                    }
                    
                    // Process RGBD frame (VO mode - no IMU support yet for RGBD)
                    est_result = estimator_->process_rgbd_frame(
                        processed_rgb,
                        dataset.depth_image,
                        timestamp_ns
                    );
                } else {
                    // ⭐ Stereo mode
                    // Preprocess images (histogram equalization)
                    cv::Mat processed_left, processed_right;
                    cv::equalizeHist(dataset.left_image, processed_left);
                    cv::equalizeHist(dataset.right_image, processed_right);
                    
                    if (processed_count == 1) {
                        // First frame: VO mode (no IMU)
                        est_result = estimator_->process_frame(
                            processed_left,
                            processed_right,
                            timestamp_ns
                        );
                    } else {
                        // Subsequent frames: VIO mode with IMU data
                        est_result = estimator_->process_frame(
                            processed_left,
                            processed_right,
                            timestamp_ns,
                            vio_imu_data
                        );
                    }
                }
                
                if (est_result.success) {
                    result.success = true;
                    result.Twb = estimator_->get_current_pose();
                    result.num_features = est_result.num_features;
                    result.num_inliers = est_result.num_inliers;
                    result.current_frame = estimator_->get_current_frame();
                    
                    // ⭐ Result queue에 push (publisher thread가 처리)
                    result_queue_.push(result);
                    
                    // RCLCPP_INFO(this->get_logger(),
                    //     "✅ VIO #%zu: features=%d, inliers=%d, opt_time=%.2fms | Queues: img=%zu, result=%zu",
                    //     processed_count, est_result.num_features, est_result.num_inliers, 
                    //     est_result.optimization_time_ms, image_queue_.size(), result_queue_.size());
                } else {
                    RCLCPP_WARN(this->get_logger(), 
                               "❌ VIO estimation failed: features=%d, inliers=%d",
                               est_result.num_features, est_result.num_inliers);
                }
                
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), 
                            "Exception in VIO processing: %s", e.what());
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "🛑 Processing thread stopped (processed %zu frames)", 
                   processed_count);
    }
    
    // ⭐ Publisher Thread: ROS topic publish (분리된 thread에서 독립 실행)
    void publisherThreadLoop() {
        RCLCPP_INFO(this->get_logger(), "📡 Publisher thread started");
        
        size_t published_count = 0;
        
        while (running_) {
            // Result queue에서 pop (blocking)
            auto result_opt = result_queue_.pop();
            
            if (!result_opt.has_value()) {
                // Queue shutdown signal
                break;
            }
            
            VIOProcessingResult result = result_opt.value();
            
            if (result.success) {
                // Publish odometry & TF
                publishOdometry(result.Twb, result.timestamp);
                
                // Publish visualization
                publishVisualization(result.timestamp);
                
                published_count++;
                
                RCLCPP_DEBUG(this->get_logger(),
                    "📤 Published frame #%zu", published_count);
            }
        }
        
        RCLCPP_INFO(this->get_logger(), 
                   "🛑 Publisher thread stopped (published %zu frames)", 
                   published_count);
    }

    void publishOdometry(const Eigen::Matrix4f& Twb, double timestamp)
    {
        // Twb: body to world transform
        // Extract rotation and translation
        Eigen::Matrix3f R = Twb.block<3, 3>(0, 0);
        Eigen::Vector3f t = Twb.block<3, 1>(0, 3);
        
        // Convert rotation matrix to quaternion
        Eigen::Quaternionf q(R);
        q.normalize();
        
        // Create ROS timestamp
        rclcpp::Time ros_time(static_cast<int64_t>(timestamp * 1e9));
        
        // Publish Odometry
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = ros_time;
        odom_msg.header.frame_id = "map";  // Fixed frame for rviz
        odom_msg.child_frame_id = "base_link";
        
        odom_msg.pose.pose.position.x = t.x();
        odom_msg.pose.pose.position.y = t.y();
        odom_msg.pose.pose.position.z = t.z();
        
        odom_msg.pose.pose.orientation.x = q.x();
        odom_msg.pose.pose.orientation.y = q.y();
        odom_msg.pose.pose.orientation.z = q.z();
        odom_msg.pose.pose.orientation.w = q.w();
        
        odom_pub_->publish(odom_msg);
        
        // Publish PoseStamped
        auto pose_msg = geometry_msgs::msg::PoseStamped();
        pose_msg.header.stamp = ros_time;
        pose_msg.header.frame_id = "map";  // Fixed frame for rviz
        
        pose_msg.pose.position.x = t.x();
        pose_msg.pose.position.y = t.y();
        pose_msg.pose.position.z = t.z();
        
        pose_msg.pose.orientation.x = q.x();
        pose_msg.pose.orientation.y = q.y();
        pose_msg.pose.orientation.z = q.z();
        pose_msg.pose.orientation.w = q.w();
        
        pose_pub_->publish(pose_msg);
        
        // Broadcast TF: map -> base_link
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = ros_time;
        transform.header.frame_id = "map";
        transform.child_frame_id = "base_link";
        
        transform.transform.translation.x = t.x();
        transform.transform.translation.y = t.y();
        transform.transform.translation.z = t.z();
        
        transform.transform.rotation.x = q.x();
        transform.transform.rotation.y = q.y();
        transform.transform.rotation.z = q.z();
        transform.transform.rotation.w = q.w();
        
        tf_broadcaster_->sendTransform(transform);
        
        // Add to trajectory (최근 5개 키프레임만 유지)
        trajectory_.poses.push_back(pose_msg);
        
        // ⭐ 최근 5개만 유지 (오래된 pose 제거)
        const size_t max_trajectory_size = 5;
        if (trajectory_.poses.size() > max_trajectory_size) {
            trajectory_.poses.erase(
                trajectory_.poses.begin(), 
                trajectory_.poses.begin() + (trajectory_.poses.size() - max_trajectory_size)
            );
        }
        
        trajectory_.header.stamp = ros_time;
        trajectory_.header.frame_id = "map";
    }
    
    void publishVisualization(double timestamp)
    {
        rclcpp::Time ros_time(static_cast<int64_t>(timestamp * 1e9));
        
        // Get current frame
        auto current_frame = estimator_->get_current_frame();
        if (!current_frame) return;
        
        // Publish current frame map points (GREEN)
        publishCurrentFramePoints(current_frame, ros_time);
        
        // Publish keyframe map points (RED)
        publishKeyframePoints(ros_time);
        
        // Publish camera frustum visualization
        publishCameraFrustum(current_frame, ros_time);
        
        // Publish tracking image with features
        publishTrackingImage(current_frame, ros_time);
        
        // ⭐ RGBD-specific publications
        if (is_rgbd_mode_ && current_frame && current_frame->is_rgbd()) {
            // Dense cloud is too slow - disabled
            // publishDenseCloud(current_frame, ros_time);
            
            // Only publish depth image (fast)
            publishDepthImage(current_frame, ros_time);
        }
        
        // Publish trajectory
        trajectory_pub_->publish(trajectory_);
    }
    
    void publishCurrentFramePoints(std::shared_ptr<Frame> frame, const rclcpp::Time& timestamp)
    {
        if (!frame) return;
        
        const auto& map_points = frame->get_map_points();
        
        // Count valid map points
        size_t valid_count = 0;
        for (const auto& mp : map_points) {
            if (mp && !mp->is_bad()) {
                valid_count++;
            }
        }
        
        if (valid_count == 0) return;
        
        // Create PointCloud2 message
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = timestamp;
        cloud_msg.header.frame_id = "map";
        cloud_msg.height = 1;
        cloud_msg.width = valid_count;
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;
        
        // Setup fields: x, y, z, rgb
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(valid_count);
        
        // Create iterators
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");
        
        // Fill point cloud (GREEN for current frame)
        for (const auto& mp : map_points) {
            if (!mp || mp->is_bad()) continue;
            
            Eigen::Vector3f pos = mp->get_position();
            *iter_x = pos.x();
            *iter_y = pos.y();
            *iter_z = pos.z();
            *iter_r = 0;
            *iter_g = 255;  // Green
            *iter_b = 0;
            
            ++iter_x; ++iter_y; ++iter_z;
            ++iter_r; ++iter_g; ++iter_b;
        }
        
        current_frame_points_pub_->publish(cloud_msg);
    }
    
    void publishKeyframePoints(const rclcpp::Time& timestamp)
    {
        // Get all keyframes (use get_keyframes_safe instead of get_all_keyframes)
        auto keyframes = estimator_->get_keyframes_safe();
        
        // Collect all unique map points from keyframes
        std::set<std::shared_ptr<MapPoint>> unique_map_points;
        for (const auto& kf : keyframes) {
            if (!kf) continue;
            const auto& map_points = kf->get_map_points();
            for (const auto& mp : map_points) {
                if (mp && !mp->is_bad()) {
                    unique_map_points.insert(mp);
                }
            }
        }
        
        if (unique_map_points.empty()) return;
        
        // Create PointCloud2 message
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = timestamp;
        cloud_msg.header.frame_id = "map";
        cloud_msg.height = 1;
        cloud_msg.width = unique_map_points.size();
        cloud_msg.is_dense = false;
        cloud_msg.is_bigendian = false;
        
        // Setup fields: x, y, z, rgb
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
        modifier.resize(unique_map_points.size());
        
        // Create iterators
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(cloud_msg, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(cloud_msg, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(cloud_msg, "b");
        
        // Fill point cloud (RED for keyframes)
        for (const auto& mp : unique_map_points) {
            Eigen::Vector3f pos = mp->get_position();
            *iter_x = pos.x();
            *iter_y = pos.y();
            *iter_z = pos.z();
            *iter_r = 255;  // Red
            *iter_g = 0;
            *iter_b = 0;
            
            ++iter_x; ++iter_y; ++iter_z;
            ++iter_r; ++iter_g; ++iter_b;
        }
        
        keyframe_points_pub_->publish(cloud_msg);
    }
    
    // ⭐ Tracking image visualization - draw features on camera image
    void publishTrackingImage(std::shared_ptr<Frame> frame, const rclcpp::Time& timestamp)
    {
        if (!frame) return;
        
        // Get left image
        const cv::Mat& raw_image = frame->get_left_image();
        if (raw_image.empty()) return;
        
        // Convert to BGR for color drawing
        cv::Mat display_image;
        if (raw_image.channels() == 1) {
            cv::cvtColor(raw_image, display_image, cv::COLOR_GRAY2BGR);
        } else {
            display_image = raw_image.clone();
        }
        
        // Get features and map points
        const auto& features = frame->get_features();
        const auto& map_points = frame->get_map_points();
        
        // Draw features with different colors
        for (size_t i = 0; i < features.size(); ++i) {
            const auto& feature = features[i];
            if (!feature || !feature->is_valid()) continue;
            
            const cv::Point2f& pt = feature->get_pixel_coord();
            
            // Determine color based on tracking status
            cv::Scalar color;
            if (frame->get_outlier_flag(i)) {
                // Gray for outliers
                color = cv::Scalar(128, 128, 128);
            } else if (i < map_points.size() && map_points[i] && !map_points[i]->is_bad()) {
                // Green for successfully tracked features (have valid map point)
                color = cv::Scalar(0, 255, 0);
            } else {
                // Red for new features (no map point yet)
                color = cv::Scalar(0, 0, 255);
            }
            
            // Draw circle at feature location
            cv::circle(display_image, pt, 3, color, -1);
        }
        
        // Convert to ROS message using cv_bridge
        std_msgs::msg::Header header;
        header.stamp = timestamp;
        header.frame_id = "camera";
        
        auto msg = cv_bridge::CvImage(header, "bgr8", display_image).toImageMsg();
        tracking_image_pub_->publish(*msg);
    }
    
    // ⭐ RGBD Dense Point Cloud Publisher
    void publishDenseCloud(std::shared_ptr<Frame> frame, const rclcpp::Time& timestamp)
    {
        if (!frame || !frame->is_rgbd()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Dense cloud: frame is null or not RGBD");
            return;
        }
        
        // ⭐ Safety check: ensure depth map and RGB image are valid
        const cv::Mat& depth_map = frame->get_depth_map();
        const cv::Mat& rgb_image = frame->get_rgb_image();
        
        if (depth_map.empty() || rgb_image.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Dense cloud: depth_map empty=%d, rgb_image empty=%d",
                                 depth_map.empty(), rgb_image.empty());
            return;
        }
        
        // Get parameters from config
        const Config& config = Config::getInstance();
        if (!config.m_rgbd_enable_dense_cloud) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "Dense cloud: disabled in config");
            return;
        }
        
        int stride = config.m_rgbd_dense_cloud_stride;
        float min_depth = config.m_min_depth;
        float max_depth = config.m_max_depth;
        
        // Generate colored point cloud (RGB mode) with try-catch for safety
        std::vector<Frame::ColoredPoint> colored_points;
        try {
            colored_points = frame->generate_colored_point_cloud(stride, min_depth, max_depth, 1);
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Failed to generate dense cloud: %s", e.what());
            return;
        }
        
        if (colored_points.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Dense cloud is empty - no valid depth points");
            return;
        }
        
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                             "Publishing dense cloud with %zu points", colored_points.size());
        
        // Create PointCloud2 message manually
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = timestamp;
        cloud_msg.header.frame_id = "map";
        cloud_msg.height = 1;
        cloud_msg.width = colored_points.size();
        cloud_msg.is_bigendian = false;
        cloud_msg.is_dense = false;
        
        // Define fields manually: x, y, z, rgb
        sensor_msgs::msg::PointField field_x, field_y, field_z, field_rgb;
        
        field_x.name = "x";
        field_x.offset = 0;
        field_x.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field_x.count = 1;
        
        field_y.name = "y";
        field_y.offset = 4;
        field_y.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field_y.count = 1;
        
        field_z.name = "z";
        field_z.offset = 8;
        field_z.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field_z.count = 1;
        
        field_rgb.name = "rgb";
        field_rgb.offset = 12;
        field_rgb.datatype = sensor_msgs::msg::PointField::FLOAT32;
        field_rgb.count = 1;
        
        cloud_msg.fields = {field_x, field_y, field_z, field_rgb};
        cloud_msg.point_step = 16; // 4 floats * 4 bytes
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width;
        cloud_msg.data.resize(cloud_msg.row_step * cloud_msg.height);
        
        // Fill point cloud data
        uint8_t* data_ptr = cloud_msg.data.data();
        for (const auto& cp : colored_points) {
            // x, y, z
            memcpy(data_ptr + 0, &cp.position.x(), sizeof(float));
            memcpy(data_ptr + 4, &cp.position.y(), sizeof(float));
            memcpy(data_ptr + 8, &cp.position.z(), sizeof(float));
            
            // Pack RGB into float (as uint32)
            uint8_t r = static_cast<uint8_t>(cp.color.x() * 255.0f);
            uint8_t g = static_cast<uint8_t>(cp.color.y() * 255.0f);
            uint8_t b = static_cast<uint8_t>(cp.color.z() * 255.0f);
            uint32_t rgb = (static_cast<uint32_t>(r) << 16) | 
                           (static_cast<uint32_t>(g) << 8) | 
                           static_cast<uint32_t>(b);
            memcpy(data_ptr + 12, &rgb, sizeof(uint32_t));
            
            data_ptr += cloud_msg.point_step;
        }
        
        dense_cloud_pub_->publish(cloud_msg);
    }
    
    // ⭐ Normalized Depth Image Publisher (heatmap visualization)
    void publishDepthImage(std::shared_ptr<Frame> frame, const rclcpp::Time& timestamp)
    {
        if (!frame || !frame->is_rgbd()) return;
        
        const cv::Mat& depth_map = frame->get_depth_map();
        if (depth_map.empty()) {
            return;
        }
        
        // ⭐ Safety check: validate depth map type
        if (depth_map.type() != CV_32FC1) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Depth map is not CV_32FC1 type");
            return;
        }
        
        // Get visualization range from config
        const Config& config = Config::getInstance();
        float min_depth = config.m_rgbd_vis_min_depth;
        float max_depth = config.m_rgbd_vis_max_depth;
        
        // Create heatmap (same as Pangolin viewer)
        cv::Mat heatmap(depth_map.rows, depth_map.cols, CV_8UC3);
        
        try {
            for (int v = 0; v < depth_map.rows; ++v) {
                for (int u = 0; u < depth_map.cols; ++u) {
                    float depth = depth_map.at<float>(v, u);
                    
                    cv::Vec3b color;
                    
                    // Depth = 0 → Black (no depth data)
                    if (depth <= 0.0f) {
                        color = cv::Vec3b(0, 0, 0);
                        heatmap.at<cv::Vec3b>(v, u) = color;
                        continue;
                    }
                    
                    // Clamp depth to [min_depth, max_depth]
                    float clamped_depth = std::max(min_depth, std::min(max_depth, depth));
                    float normalized_depth = (clamped_depth - min_depth) / (max_depth - min_depth);
                    normalized_depth = std::max(0.0f, std::min(1.0f, normalized_depth));
                    
                    // Heatmap: Red (near) -> Yellow -> Green -> Cyan -> Blue (far)
                    float r, g, b;
                    if (normalized_depth < 0.25f) {
                        float t = normalized_depth / 0.25f;
                        r = 1.0f; g = t; b = 0.0f;
                    } else if (normalized_depth < 0.5f) {
                        float t = (normalized_depth - 0.25f) / 0.25f;
                        r = 1.0f - t; g = 1.0f; b = 0.0f;
                    } else if (normalized_depth < 0.75f) {
                        float t = (normalized_depth - 0.5f) / 0.25f;
                        r = 0.0f; g = 1.0f; b = t;
                    } else {
                        float t = (normalized_depth - 0.75f) / 0.25f;
                        r = 0.0f; g = 1.0f - t; b = 1.0f;
                    }
                    
                    // Convert to BGR for OpenCV
                    color = cv::Vec3b(
                        static_cast<uchar>(b * 255),
                        static_cast<uchar>(g * 255),
                        static_cast<uchar>(r * 255)
                    );
                    
                    heatmap.at<cv::Vec3b>(v, u) = color;
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                 "Failed to create depth heatmap: %s", e.what());
            return;
        }
        
        // Convert to ROS message
        std_msgs::msg::Header header;
        header.stamp = timestamp;
        header.frame_id = "camera";
        
        auto msg = cv_bridge::CvImage(header, "bgr8", heatmap).toImageMsg();
        depth_image_pub_->publish(*msg);
    }
    
    // ⭐ Camera frustum visualization - LINE_LIST로 pyramid 모양 그리기
    void publishCameraFrustum(std::shared_ptr<Frame> frame, const rclcpp::Time& timestamp)
    {
        if (!frame) return;
        
        // Get camera pose: Twc (camera to world)
        Eigen::Matrix4f Twc = frame->get_Twc();
        Eigen::Vector3f camera_pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f R_wc = Twc.block<3, 3>(0, 0);
        
        // ⭐ Get camera intrinsics from frame
        float fx = frame->get_fx();
        float fy = frame->get_fy();
        float cx = frame->get_cx();
        float cy = frame->get_cy();
        
        // ⭐ Get actual image size from frame
        const cv::Mat& img = frame->get_left_image();
        float img_width = static_cast<float>(img.cols);
        float img_height = static_cast<float>(img.rows);
        
        // Frustum size (depth in meters from camera center)
        float frustum_depth = 0.2f;  // 20cm pyramid depth
        
        // ⭐ Calculate 4 corners in normalized image coordinates, then scale by depth
        // Corners: Top-left, Top-right, Bottom-right, Bottom-left
        Eigen::Vector3f corners_cam[4];
        
        // Top-left (0, 0)
        corners_cam[0] = Eigen::Vector3f((0.0f - cx) / fx, (0.0f - cy) / fy, 1.0f);
        corners_cam[0] = corners_cam[0].normalized() * frustum_depth;
        
        // Top-right (width, 0)
        corners_cam[1] = Eigen::Vector3f((img_width - cx) / fx, (0.0f - cy) / fy, 1.0f);
        corners_cam[1] = corners_cam[1].normalized() * frustum_depth;
        
        // Bottom-right (width, height)
        corners_cam[2] = Eigen::Vector3f((img_width - cx) / fx, (img_height - cy) / fy, 1.0f);
        corners_cam[2] = corners_cam[2].normalized() * frustum_depth;
        
        // Bottom-left (0, height)
        corners_cam[3] = Eigen::Vector3f((0.0f - cx) / fx, (img_height - cy) / fy, 1.0f);
        corners_cam[3] = corners_cam[3].normalized() * frustum_depth;
        
        // Transform corners to world frame
        Eigen::Vector3f corners_world[4];
        for (int i = 0; i < 4; ++i) {
            corners_world[i] = R_wc * corners_cam[i] + camera_pos;
        }
        
        // Create LINE_LIST marker
        visualization_msgs::msg::Marker frustum_marker;
        frustum_marker.header.stamp = timestamp;
        frustum_marker.header.frame_id = "map";
        frustum_marker.ns = "camera_frustum";
        frustum_marker.id = 0;
        frustum_marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        frustum_marker.action = visualization_msgs::msg::Marker::ADD;
        
        // Frustum color (cyan)
        frustum_marker.color.r = 0.0f;
        frustum_marker.color.g = 1.0f;
        frustum_marker.color.b = 1.0f;
        frustum_marker.color.a = 1.0f;
        
        // Line width
        frustum_marker.scale.x = 0.01;  // 1cm thick lines
        
        // Add lines: camera center to 4 corners
        for (int i = 0; i < 4; ++i) {
            geometry_msgs::msg::Point p_center, p_corner;
            p_center.x = camera_pos.x();
            p_center.y = camera_pos.y();
            p_center.z = camera_pos.z();
            
            p_corner.x = corners_world[i].x();
            p_corner.y = corners_world[i].y();
            p_corner.z = corners_world[i].z();
            
            frustum_marker.points.push_back(p_center);
            frustum_marker.points.push_back(p_corner);
        }
        
        // Add lines connecting 4 corners (rectangle)
        for (int i = 0; i < 4; ++i) {
            geometry_msgs::msg::Point p1, p2;
            
            p1.x = corners_world[i].x();
            p1.y = corners_world[i].y();
            p1.z = corners_world[i].z();
            
            int next = (i + 1) % 4;
            p2.x = corners_world[next].x();
            p2.y = corners_world[next].y();
            p2.z = corners_world[next].z();
            
            frustum_marker.points.push_back(p1);
            frustum_marker.points.push_back(p2);
        }
        
        camera_frustum_pub_->publish(frustum_marker);
    }

    // Subscribers
    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;

    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr current_frame_points_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr keyframe_points_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr camera_frustum_pub_;  // ⭐ Camera frustum publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr tracking_image_pub_;  // ⭐ Tracking image publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dense_cloud_pub_;  // ⭐ RGBD dense point cloud publisher
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_pub_;  // ⭐ Normalized depth image publisher
    
    // TF broadcaster
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    
    // Trajectory storage
    nav_msgs::msg::Path trajectory_;

    // Synchronizer for stereo images
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, 
            sensor_msgs::msg::Image>>> sync_;
    
    // Synchronizer for RGBD images
    std::shared_ptr<message_filters::Synchronizer<
        message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::Image, 
            sensor_msgs::msg::Image>>> rgbd_sync_;

    // IMU buffer and pivot
    std::vector<ImuData> imu_buffer_;
    size_t imu_pivot_index_;
    double last_image_time_;
    double imu_time_tolerance_;

    // Counters
    size_t stereo_count_;
    size_t rgbd_count_;
    size_t imu_count_;
    size_t left_count_;
    size_t right_count_;
    size_t dataset_id_;
    
    // Mode flag
    bool is_rgbd_mode_;

    // Timestamps
    double first_stereo_time_;
    double last_stereo_time_;
    double first_rgbd_time_;
    double last_rgbd_time_;
    double first_imu_time_;
    double last_imu_time_;

    // VIO Estimator
    std::shared_ptr<Estimator> estimator_;
    
    // ⭐ Producer-Consumer Pattern: Thread-safe Queues
    ThreadSafeQueue<VioDataset> image_queue_;           // Image callback → Processing thread
    ThreadSafeQueue<VIOProcessingResult> result_queue_; // Processing thread → Publisher thread
    
    // ⭐ Worker Threads
    std::thread processing_thread_;  // VIO processing (heavy computation)
    std::thread publisher_thread_;   // ROS publishing (isolated from processing)
    std::atomic<bool> running_;      // Thread control flag
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VIONode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
