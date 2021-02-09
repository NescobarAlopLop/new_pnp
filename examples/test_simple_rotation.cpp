//
// Created by george on 2/8/21.
//

#include "../PnPProblem.h"
#include <PnpProblemSolver.h>
#include <iostream>
//#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cxeigen.hpp>


using namespace cv;


int main()
{
	int x1 = 20, y1 = 30;
	int x2 = 50, y2 = 30;
	int x3 = 50, y3 = 60;
	int x4 = 20, y4 = 60;
	std::vector<cv::Point3f> list_points3d_model_match;
	int distance1 = 400;
	list_points3d_model_match.emplace_back(cv::Point3f(x1,y1, distance1));
	list_points3d_model_match.emplace_back(cv::Point3f(x2,y2, distance1));
	list_points3d_model_match.emplace_back(cv::Point3f(x3,y3, distance1));
	list_points3d_model_match.emplace_back(cv::Point3f(x4,y4, distance1));

	int distance2 = 500;
	list_points3d_model_match.emplace_back(cv::Point3f(x1,y1, distance2));
	list_points3d_model_match.emplace_back(cv::Point3f(x2,y2, distance2));
	list_points3d_model_match.emplace_back(cv::Point3f(x3,y3, distance2));
	list_points3d_model_match.emplace_back(cv::Point3f(x4,y4, distance2));

	int distance3 = 600;
	list_points3d_model_match.emplace_back(cv::Point3f(x1,y1, distance3));
	list_points3d_model_match.emplace_back(cv::Point3f(x2,y2, distance3));
	list_points3d_model_match.emplace_back(cv::Point3f(x3,y3, distance3));
	list_points3d_model_match.emplace_back(cv::Point3f(x4,y4, distance3));

	int distance4 = 800;
	list_points3d_model_match.emplace_back(cv::Point3f(x1,y1, distance4));
	list_points3d_model_match.emplace_back(cv::Point3f(x2,y2, distance4));
	list_points3d_model_match.emplace_back(cv::Point3f(x3,y3, distance4));
	list_points3d_model_match.emplace_back(cv::Point3f(x4,y4, distance4));

	int distance5 = 1200;
	list_points3d_model_match.emplace_back(cv::Point3f(x1,y1, distance5));
	list_points3d_model_match.emplace_back(cv::Point3f(x2,y2, distance5));
	list_points3d_model_match.emplace_back(cv::Point3f(x3,y3, distance5));
	list_points3d_model_match.emplace_back(cv::Point3f(x4,y4, distance5));

	Mat rvec, tvec;
	rvec.create(1, 3, CV_32F);
	rvec.at<float>(0) = 0.2f;//.785398;  // 45 degrees = 0.785398 Rad
	rvec.at<float>(1) = 0.0785398f;
	rvec.at<float>(2) = 0.5f;//.785398;
	Matx33f original_rotation_matrix;
	cv::Rodrigues(rvec, original_rotation_matrix, noArray());

	tvec.create(3, 1, CV_32F);
	tvec.at<float>(0) = -100;
	tvec.at<float>(1) = 20;
	tvec.at<float>(2) = -300;

	std::vector<Point3f> world;
	for (const auto& point3d:list_points3d_model_match)
	{
		world.push_back(point3d);
	}

	// Intrinsic camera parameters: UVC WEBCAM
	float f = 24;                           // focal length in mm
	float sx = 22.3, sy = 14.9;             // sensor size
	int width = 640, height = 480;        // image size

	float fx = (float)width*f/sx;
	float fy = (float)height*f/sy;
	float cx = (float)width/2;
	float cy = (float)height/2;

	double params_WEBCAM[] = { fx,
							   fy,
							   cx,
							   cy};
	Mat cmat;
//	cmat.create(3, 3, CV_32F);
	cmat.create(3, 3, CV_32F);
	setIdentity(cmat);
	cmat.at<float>(0, 0) = (float)fx;
	cmat.at<float>(1, 1) = (float)fy;
	cmat.at<float>(0, 2) = (float)cx;
	cmat.at<float>(1, 2) = (float)cy;

	std::vector<Point2f> image;
	projectPoints(world, rvec, tvec, cmat, noArray(), image);

	char atom_window[] = "Drawing 1: Atom";
	Mat atom_image = Mat::zeros(height, width, CV_8UC3);
	circle(atom_image, Point (1,1), 1, Scalar(20, 255, 0), 4);

	std::vector<cv::Point2f> list_points2d_scene_match;
	for (const auto& point:image)
	{
		circle(atom_image, point, 1, CV_RGB(200, 255, 220), 3);
		list_points2d_scene_match.emplace_back(point);
	}

	cv::Mat inliers_idx;
	PnPProblem pnp_detection(params_WEBCAM); // instantiate PnPProblem class

	// RANSAC parameters
	int iterationsCount = 500;        // number of Ransac iterations.
	float reprojectionError = 2.0;    // maximum allowed distance to consider it an inlier.
	float confidence = 0.95;          // ransac successful confidence.
	int pnpMethod = cv::SOLVEPNP_ITERATIVE;
	pnp_detection.estimatePoseRANSAC( list_points3d_model_match, list_points2d_scene_match,
		pnpMethod, inliers_idx, iterationsCount, reprojectionError, confidence);
	auto pnp_rotation_matrix = pnp_detection.get_R_matrix();
	auto pnp_translation_vector = pnp_detection.get_t_matrix();



	std::vector<Eigen::Vector3d> eigen_points;
	for(const auto& point:list_points3d_model_match)
	{
		eigen_points.emplace_back((double)point.x, (double)point.y, (double)point.z);
	}


	std::vector<Eigen::Vector3d> eigen_lines;
	for (const auto& iter_point: list_points2d_scene_match)
	{
		cv::Mat point = (cv::Mat_<float>(3, 1) << iter_point.x, iter_point.y, 1);
		cv::Mat temp_line = cmat.inv() * point;
		temp_line = temp_line / norm(temp_line);
		cv::Point3d temp_point3d(temp_line);

		eigen_lines.emplace_back((double)temp_point3d.x, (double)temp_point3d.y, (double)temp_point3d.z);
	}

	std::vector<double> weights;
	std::vector<int> indices;
	for (int i=0; i < eigen_points.size(); ++i)
	{
		weights.push_back(1);
		indices.push_back(i);
	}

	auto pnp = PnP::PnpProblemSolver::init();
	auto barrier_method_settings = PnP::BarrierMethodSettings::init();
	barrier_method_settings->epsilon = 4E-8;
	barrier_method_settings->verbose = false;

	auto pnp_input = PnP::PnpInput::init(eigen_points, eigen_lines, weights, indices);
	auto pnp_objective = PnP::PnpObjective::init(pnp_input);
	auto pnp_res = pnp->solve_pnp(pnp_objective, barrier_method_settings);

	auto new_rotationMatrix = pnp_res.rotation_matrix();
	auto translationVector = pnp_res.translation_vector();

	cv::Mat cvR, cvt;
	cv::eigen2cv(new_rotationMatrix.transpose().eval(),cvR);
	cv::eigen2cv(translationVector,cvt);

	std::cout << "original_rotation_matrix" << "\t\t\t\t|\t" << "pnp_rotation_matrix" << "\t\t\t\t\t\t\t\t\t\t\t\t|\t" << "new_rotationMatrix" << std::endl;
	for(int row=0; row < original_rotation_matrix.rows; ++row)
	{
		std::cout << original_rotation_matrix.row(row) << "  \t|\t" << pnp_rotation_matrix.row(row) << " \t|\t" << cvR.row(row) << std::endl;
	}

	std::cout << std::endl << std::endl << "original_translation_vector" << "\t|\t" << "pnp translation vector" << "\t|\t" << "new translation vector" << std::endl;
	for(int row=0; row < tvec.rows; ++row)
	{
		std::cout << tvec.row(row) << " \t\t\t\t\t\t|\t" << pnp_translation_vector.row(row) << " \t|\t" << cvt.row(row) << std::endl;
	}

	std::cout << "cost:" << std::endl << pnp_res.cost() << std::endl;

	imshow(atom_window, atom_image);
//	waitKey(0);

	return 0;
}
