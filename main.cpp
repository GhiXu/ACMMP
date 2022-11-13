#include "main.h"
#include "ACMMP.h"

void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");

    problems.clear();

    std::ifstream file(cluster_list_path);

    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

int ComputeMultiScaleSettings(const std::string &dense_folder, std::vector<Problem> &problems)
{
    int max_num_downscale = -1;
    int size_bound = 1000;
    PatchMatchParams pmp;
    std::string image_folder = dense_folder + std::string("/images");

    size_t num_images = problems.size();

    for (size_t i = 0; i < num_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);

        int rows = image_uint.rows;
        int cols = image_uint.cols;
        int max_size = std::max(rows, cols);
        if (max_size > pmp.max_image_size) {
            max_size = pmp.max_image_size;
        }
        problems[i].max_image_size = max_size;

        int k = 0;
        while (max_size > size_bound) {
            max_size /= 2;
            k++;
        }

        if (k > max_num_downscale) {
            max_num_downscale = k;
        }

        problems[i].num_downscale = k;
    }

    return max_num_downscale;
}

void ProcessProblem(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx, bool geom_consistency, bool planar_prior, bool hierarchy, bool multi_geometrty=false)
{
    const Problem problem = problems[idx];
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(0);
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    ACMMP acmmp;
    if (geom_consistency) {
        acmmp.SetGeomConsistencyParams(multi_geometrty);
    }
    if (hierarchy) {
        acmmp.SetHierarchyParams();
    }

    acmmp.InuputInitialization(dense_folder, problems, idx);

    acmmp.CudaSpaceInitialization(dense_folder, problem);
    acmmp.RunPatchMatch();

    const int width = acmmp.GetReferenceImageWidth();
    const int height = acmmp.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmmp.GetCost(center);
        }
    }

    if (planar_prior) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        acmmp.SetPlanarPriorParams();

        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;

        acmmp.GetSupportPoints(support2DPoints);
        const auto triangles = acmmp.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = acmmp.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0; // To distinguish from the label of non-triangulated areas
                    }
                }

                // estimate plane parameter
                float4 n4 = acmmp.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = acmmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= acmmp.GetMaxDepth() && d >= acmmp.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }
        // std::string depth_path = result_folder + "/depths_prior.dmb";
        //  writeDepthDmb(depth_path, priordepths);

        acmmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        acmmp.RunPatchMatch();

        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = acmmp.GetCost(center);
            }
        }
    }

    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void JointBilateralUpsampling(const std::string &dense_folder, const Problem &problem, int acmmp_size)
{
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths_geom.dmb";
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);

    std::string image_folder = dense_folder + std::string("/images");
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    const float factor_x = static_cast<float>(acmmp_size) / image_float.cols;
    const float factor_y = static_cast<float>(acmmp_size) / image_float.rows;
    const float factor = std::min(factor_x, factor_y);

    const int new_cols = std::round(image_float.cols * factor);
    const int new_rows = std::round(image_float.rows * factor);
    cv::Mat scaled_image_float;
    cv::resize(image_float, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);

    std::cout << "Run JBU for image " << problem.ref_image_id <<  ".jpg" << std::endl;
    RunJBU(scaled_image_float, ref_depth, dense_folder, problem );
}

void RunFusion(std::string &dense_folder, const std::vector<Problem> &problems, bool geom_consistency)
{
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();
    
    std::map<int, int> image_id_2_index;

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        image_id_2_index[problems[i].ref_image_id] = i;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = problems[i].src_image_ids.size();
        std::vector<int2> used_list(num_ngb, make_int2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                float3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                float3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;
                float dynamic_consistency = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = image_id_2_index[problems[i].src_image_ids[j]];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    float2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;

                        float3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        float2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                           /* consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];*/
                           

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;

                            float tmp_index = reproj_error + 200 * relative_depth_diff + angle * 10;
                            float cons = exp(-tmp_index);
                            dynamic_consistency += exp(-tmp_index);
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 1 && (dynamic_consistency > 0.3 * num_consistent)) {
                    /*consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);*/

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = make_float3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = make_float3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[image_id_2_index[problems[i].src_image_ids[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = dense_folder + "/ACMMP/ACMMP_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "USAGE: ACMMP dense_folder" << std::endl;
        return -1;
    }

    std::string dense_folder = argv[1];
    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);

    std::string output_folder = dense_folder + std::string("/ACMMP");
    mkdir(output_folder.c_str(), 0777);

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems);

     int flag = 0;
     int geom_iterations = 2;
     bool geom_consistency = false;
     bool planar_prior = false;
     bool hierarchy = false;
     bool multi_geometry = false;
     while (max_num_downscale >= 0) {
        std::cout << "Scale: " << max_num_downscale << std::endl;

        for (size_t i = 0; i < num_images; ++i) {
            if (problems[i].num_downscale >= 0) {
                problems[i].cur_image_size = problems[i].max_image_size / pow(2, problems[i].num_downscale);
                problems[i].num_downscale--;
            }
        }

        if (flag == 0) {
            flag = 1;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i) {
                ProcessProblem(dense_folder, problems, i, geom_consistency, planar_prior, hierarchy);
            }
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                if (geom_iter == 0) {
                    multi_geometry = false;
                }
                else {
                    multi_geometry = true;
                }
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy, multi_geometry);
                }
            }
        }
        else {
            for (size_t i = 0; i < num_images; ++i) {
               JointBilateralUpsampling(dense_folder, problems[i], problems[i].cur_image_size);
             }

            hierarchy = true;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i) {
                ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy);
            }
            hierarchy = false;
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                if (geom_iter == 0) {
                    multi_geometry = false;
                }
                else {
                    multi_geometry = true;
                }
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(dense_folder,  problems, i, geom_consistency, planar_prior, hierarchy, multi_geometry);
                }
            }
        }

        max_num_downscale--;
    }

    geom_consistency = true;
    RunFusion(dense_folder, problems, geom_consistency);

    return 0;
}
