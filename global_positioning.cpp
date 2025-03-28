#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

/*
global positioningに必要なデータ
    方向ベクトル（観測値からの算出）
        global rotation
        camera parameter
        画像内における特徴点のピクセル位置
        　→本来ならロードしているが簡単のためtrackの度乱数生成
        　→というか今回はTrackが略式
    特徴点の3D位置（変数）
    カメラの3D位置（変数）
*/

struct BATAPairwiseDirectionError
{
    BATAPairwiseDirectionError(const Eigen::Vector3d& translation_obs)
        : translation_obs_(translation_obs) {}
  
    // The error is given by the position error described above.
    template <typename T>
    bool operator()(const T* position1,
                    const T* position2,
                    const T* scale,
                    T* residuals) const {
      Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vec(residuals);
      residuals_vec =
          translation_obs_.cast<T>() -
          scale[0] * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position2) -
                      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(position1));
      return true;
    }
  
    static ceres::CostFunction* Create(const Eigen::Vector3d& translation_obs) {
      return (
          new ceres::AutoDiffCostFunction<BATAPairwiseDirectionError, 3, 3, 3, 1>(
              new BATAPairwiseDirectionError(translation_obs)));
    }
  
    // TODO: add covariance
    const Eigen::Vector3d translation_obs_;
};

struct CameraIntrinsics
{
    double fx = 800, fy = 800;
    double cx = 640, cy = 480;

    cv::Mat K() const
    {
        Eigen::Matrix3d matEigen = Eigen::Matrix3d::Zero();
        matEigen(0, 0) = fx;
        matEigen(1, 1) = fy;
        matEigen(0, 2) = cx;
        matEigen(1, 2) = cy;
        matEigen(2, 2) = 1.0;
        
        cv::Mat mat;
        cv::eigen2cv(matEigen, mat);
        
        return mat;
    }

    cv::Mat Kinv() const
    {
        return K().inv();
    }
};

struct MyTrack
{
    std::unordered_set<int> cameraIndice;
};

Eigen::Vector3d rndVector3d(std::mt19937& rng, double low, double high)
{
    std::uniform_real_distribution<double> distribution(low, high);
    return 100.0 * Eigen::Vector3d(
        distribution(rng), distribution(rng), distribution(rng)
    );
}

cv::Matx33d RndRotationMatrix(std::mt19937& rng, double low, double high)
{
    // low: 0.0, high: 1.0
    std::uniform_real_distribution<double> distribution(low, high);
    double u1 = distribution(rng), u2 = distribution(rng), u3 = distribution(rng);
  
    double w = std::sqrt(1 - u1) * std::sin(2 * M_PI * u2);
    double x = std::sqrt(1 - u1) * std::cos(2 * M_PI * u2);
    double y = std::sqrt(u1) * std::sin(2 * M_PI * u3);
    double z = std::sqrt(u1) * std::cos(2 * M_PI * u3);
  
    Eigen::Quaterniond q(w, x, y, z);
    Eigen::Matrix3d matEigen = q.normalized().toRotationMatrix();
    
    cv::Matx33d mat;
    cv::eigen2cv(matEigen, mat);
    
    return mat;
}

void generateTracks(std::mt19937& rng, std::vector<MyTrack>& tracks, int ntracks, int ncameras, int minObserver, int maxObserver)
{
    std::uniform_int_distribution<int> distribution(0, ncameras-1);
    std::uniform_int_distribution<int> obsDistribution(minObserver, maxObserver);

    for (int i = 0; i < ntracks; i++)
    {
        int nobserver = obsDistribution(rng);
        MyTrack t = MyTrack();
        while (t.cameraIndice.size() < nobserver)
        {
            t.cameraIndice.insert(distribution(rng));
        };
        tracks.emplace_back(t);
    }
}

void globalPositioning(std::mt19937& rng, ceres::Problem* problem, ceres::Solver::Options solverOptions, 
    std::vector<double>& scales, std::vector<MyTrack>& tracks, std::vector<cv::Matx33d>& globalRotations,
    std::vector<Eigen::Vector3d>& globalTranslations, std::vector<Eigen::Vector3d>& points3D, CameraIntrinsics& params)
{
    ceres::ParameterBlockOrdering* ordering = new ceres::ParameterBlockOrdering();
    solverOptions.linear_solver_ordering.reset(ordering);

    std::uniform_int_distribution<int> xdistribution(0, 640);
    std::uniform_int_distribution<int> ydistribution(0, 480);

    for (size_t i = 0; i < tracks.size(); i++)
    {
        MyTrack t = tracks[i];
        for (auto itr = t.cameraIndice.begin(); itr != t.cameraIndice.end(); itr++)
        {
            int camIdx = *itr;
            double& scale = scales.emplace_back(1.0);

            if (scale < 1e-6) {
                std::cerr << "Warning: scale close to 0, may be degenerate residual." << std::endl;
            }

            cv::Matx33d globalRotation = globalRotations[camIdx];
            
            int ptx = xdistribution(rng), pty = ydistribution(rng);
            cv::Vec3d vHomo = cv::Vec3d(ptx, pty, 1);
            cv::Mat _vInCam = params.Kinv() * vHomo;
            cv::Vec3d vUnit = cv::Vec3d(_vInCam.at<double>(0), _vInCam.at<double>(1), _vInCam.at<double>(2));
            cv::Vec3d v = globalRotation.inv() * vUnit;
            Eigen::Vector3d vEigen = Eigen::Vector3d(v[0], v[1], v[2]);

            if (!vEigen.allFinite()) {
                std::cerr << "Invalid observation vector (NaN or Inf), skipping." << std::endl;
                continue;
            }

            Eigen::Vector3d& point3D = points3D[i];
            Eigen::Vector3d& globalTranslation = globalTranslations[camIdx];

            ceres::CostFunction* costFunction = BATAPairwiseDirectionError::Create(vEigen);

            problem->AddParameterBlock(&scale, 1);
			problem->SetParameterLowerBound(&scale, 0, 1e-5);

            ordering->AddElementToGroup(&scale, 0);
            ordering->AddElementToGroup(point3D.data(), 1);
            ordering->AddElementToGroup(globalTranslation.data(), 2);

            problem->AddResidualBlock(costFunction, nullptr, globalTranslation.data(), point3D.data(), &scale);
        }
    }
}

void saveVectorsToFile(const std::string& filepath, const std::vector<Eigen::Vector3d>& vectors)
{
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filepath << std::endl;
        return;
    }
    
    for (const auto& vec : vectors) {
        file << vec.x() << " " << vec.y() << " " << vec.z() << "\n";
    }
    
    file.close();
}

void saveSummaryToFile(const std::string& filepath, const ceres::Solver::Summary summary)
{
    std::ofstream out(filepath);
    if (out.is_open()) {
        out << summary.FullReport() << std::endl;
        out.close();
    } else {
        std::cerr << "Failed to open file: " << filepath << std::endl;
    }
}

int main(int argc, char* argv[])
{
    const std::string saveDir = "../work/";
    const std::string ext = ".txt";

    int seed = 1234;
    int ncameras = 23;
    int ntracks = 100000;
    int minObserver = 3;
    int maxObserver = 7;

    std::mt19937 rng;
    rng.seed(seed);

    if (!std::filesystem::exists(saveDir)) std::filesystem::create_directories(saveDir);

    std::vector<cv::Matx33d> globalRotations(ncameras);
    std::generate(globalRotations.begin(), globalRotations.end(),
        [&]() { return RndRotationMatrix(rng, 0.0, 1.0); });

    std::vector<Eigen::Vector3d> globalTranslations(ncameras);
    std::generate(globalTranslations.begin(), globalTranslations.end(),
        [&]() { return rndVector3d(rng, -1.0, 1.0); });
    std::vector<Eigen::Vector3d> points3D(ntracks);
    std::generate(points3D.begin(), points3D.end(),
        [&]() { return rndVector3d(rng, -1.0, 1.0); });

    saveVectorsToFile(saveDir + "bf_gloT" + ext, globalTranslations);
    saveVectorsToFile(saveDir + "bf_pnt" + ext, points3D);

    CameraIntrinsics params = CameraIntrinsics();

    std::vector<MyTrack> tracks;
    generateTracks(rng, tracks, ntracks, ncameras, minObserver, maxObserver);

    std::vector<double> scales;
    scales.reserve(tracks.size() * maxObserver);

    ceres::Problem::Options problemOptions;
    problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    std::unique_ptr<ceres::Problem> problem = std::make_unique<ceres::Problem>(problemOptions);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.preconditioner_type = ceres::CLUSTER_TRIDIAGONAL;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;

    globalPositioning(rng, problem.get(), options, scales, tracks, globalRotations, globalTranslations, points3D, params);

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem.get(), &summary);
    std::cout << summary.BriefReport() << std::endl;

    saveVectorsToFile(saveDir + "af_gloT" + ext, globalTranslations);
    saveVectorsToFile(saveDir + "af_pnt" + ext, points3D);
    saveSummaryToFile(saveDir + "ceres_summary" + ext, summary);
}
