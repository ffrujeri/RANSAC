//
// Created by Felipe Vieira Frujeri on 29/12/15.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <random>

#include "include/Ransac.h"
#include "LineModel.h"

RANSAC::VPFloat computeSlope(int x0, int y0, int x1, int y1) {
    return (RANSAC::VPFloat) (y1-y0) / (x1-x0);
}

void DrawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int lineWidth) {
    RANSAC::VPFloat slope = computeSlope(a.x, a.y, b.x, b.y);
    cv::Point p(0,0), q(img.cols, img.rows);

    p.y = -(a.x - p.x) * slope + a.y;
    q.y = -(b.x - q.x) * slope + b.y;

    cv::line(img, p, q, color, lineWidth, 8, 0);
}

int main(int argc, char* argv[]) {
    if (argc != 1 && argc != 3) {
        std::cout << "[ USAGE ]: " << argv[0] << " [<Image Size> = 1000] [<nPoints> = 500]" << std::endl;
        return -1;
    }

    int side = 500;
    int nPoints = 500;
    if (argc == 3) {
        side = std::atoi(argv[1]);
        nPoints = std::atoi(argv[2]);
    }
    cv::Mat canvas(side, side, CV_8UC3);
    canvas.setTo(255);

    std::random_device seedDevice;
    std::mt19937 RNG = std::mt19937(seedDevice());

    std::uniform_int_distribution<int> uniDist(0, side-1);

    int perturb = 50;
    std::normal_distribution<RANSAC::VPFloat> perturbDist(0, perturb);

    std::vector<std::shared_ptr<RANSAC::Parameter> > candPoints;

    for (int i = 0; i < nPoints; i++) {
        int diag = uniDist(RNG);
        cv::Point pt((int) floor(side - diag + perturbDist(RNG)), (int) floor(diag + perturbDist(RNG)));
        cv::circle(canvas, pt, int(floor(side / 200)), cv::Scalar(0, 0, 255), -1);

        std::shared_ptr<RANSAC::Parameter> candPt = std::make_shared<Point2D>(pt.x, pt.y);
        candPoints.push_back(candPt);
    }

    RANSAC::Estimator<Line2DModel, 2> estimator;
    estimator.initialize(20,100);
    int start = cv::getTickCount();
    estimator.estimate(candPoints);
    int end = cv::getTickCount();
    std::cout << "RANSAC took: " << RANSAC::VPFloat(end-start) / RANSAC::VPFloat(cv::getTickFrequency()) * 1000.0 << " ms." << std::endl;

    auto bestInliers = estimator.getBestInliers();
    if (bestInliers.size() > 0) {
        for (auto& inlier : bestInliers) {
            auto rPt = std::dynamic_pointer_cast<Point2D>(inlier);
            cv::Point pt((int)floor(rPt->point2D[0]), (int) floor(rPt->point2D[1]));
            cv::circle(canvas, pt, int(floor(side/200)), cv::Scalar(0, 255, 0), -1);
        }

        auto bestLine = estimator.getBestModel();
        if(bestLine) {
            auto bestLinePt1 = std::dynamic_pointer_cast<Point2D>(bestLine->getModelParams()[0]);
            auto bestLinePt2 = std::dynamic_pointer_cast<Point2D>(bestLine->getModelParams()[1]);
            if (bestLinePt1 && bestLinePt2) {
                cv::Point pt1((int)bestLinePt1->point2D[0], (int)bestLinePt1->point2D[1]);
                cv::Point pt2((int)bestLinePt2->point2D[0], (int)bestLinePt2->point2D[1]);
                DrawFullLine(canvas, pt1, pt2, cv::Scalar(0,0,0), 2);
            }
        }
        while(true) {
            cv::imshow("RANSAC example", canvas);

            char key = cv::waitKey(1);
            if (key == 27)
                return 0;
            if(key == ' ')
                cv::imwrite("LineFitting.png", canvas);
        }
        return 0;
    }

}
