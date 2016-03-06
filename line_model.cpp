#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>

#include "ransac.hpp"

// ---------------------------------------------------------------
// 2D line model implementation: Point2D and Line2DModel
class Point2D : public RANSAC::Parameter {
public:
  const double x, y;
  Point2D(double x, double y) : x(x), y(y) {};
};

class Line2DModel : public RANSAC::Model<2> {
public:
  static const int MODEL_SIZE = 2;

  Line2DModel(const std::vector<RANSAC::Parameter*>& inputParameters) {
    initialize(inputParameters);
  }

  virtual void initialize(const std::vector<RANSAC::Parameter*>& inputParameters) override {
    if (inputParameters.size() != MODEL_SIZE) {
      throw std::runtime_error("Line2DModel - wrong size for input parameters");
    }

    Point2D* point1 = dynamic_cast<Point2D*>(inputParameters[0]);
    Point2D* point2 = dynamic_cast<Point2D*>(inputParameters[1]);
    if (point1 == NULL || point2 == NULL) {
      throw std::runtime_error("Line2DModel - parameter mismatch (expected Point2D)");
    }

    std::copy(inputParameters.begin(), inputParameters.end(), minModelParameters.begin());
    a = (point2->y - point1->y) / (point2->x - point1->x);
    b = -1.0;
    c = point1->y - a * point1->x;
  }

  virtual std::pair<double, std::vector<RANSAC::Parameter*> > evaluate(
    std::vector<RANSAC::Parameter*> evaluateParameters, double threshold) override {
    std::vector<RANSAC::Parameter*> inliers;
    int nTotalParams = int(evaluateParameters.size());
    int nInliers = 0;

    for (RANSAC::Parameter*& parameter : evaluateParameters) {
      if (computeDistance(parameter) < threshold) {
        inliers.push_back(parameter);
        nInliers++;
      }
    }

    double inliersFraction = double(nInliers) / double(nTotalParams);
    return std::make_pair(inliersFraction, inliers);
  }

protected:
  virtual double computeDistance(RANSAC::Parameter* parameter) override {
    Point2D* point = dynamic_cast<Point2D*>(parameter);
    return fabs(a * point->x + b * point->y + c) / sqrt(a*a + b*b);
  };

private:
  double a, b, c;
};

// ---------------------------------------------------------------
// functions used in main: generate data, draw and show results
std::vector<RANSAC::Parameter*> generateData(int dataSize, int interval) {
  std::vector<RANSAC::Parameter*> candidatePoints;
  std::random_device seedDevice;
  std::mt19937 RNG = std::mt19937(seedDevice());
  std::uniform_int_distribution<int> uniDist(0, interval - 1);
  int perturb = 50;
  std::normal_distribution<double> perturbDist(0, perturb);
  for (int i = 0; i < dataSize; i++) {
    int diag = uniDist(RNG);
    double x = floor(interval - diag + perturbDist(RNG));
    double y = floor(diag + perturbDist(RNG));
    Point2D* point = new Point2D(x, y);
    candidatePoints.push_back(point);
  }

  return candidatePoints;
}

double computeSlope(int x0, int y0, int x1, int y1) {
  return double(y1 - y0) / double(x1 - x0);
}

void drawFullLine(cv::Mat& img, cv::Point a, cv::Point b, cv::Scalar color, int lineWidth) {
  double slope = computeSlope(a.x, a.y, b.x, b.y);
  cv::Point p(0, 0), q(img.cols, img.rows);
  p.y = int(-(a.x - p.x) * slope + a.y);
  q.y = int(-(b.x - q.x) * slope + b.y);
  cv::line(img, p, q, color, lineWidth, 8, 0);
}

void drawPoints(
    const std::vector<RANSAC::Parameter*>& points, cv::Mat& canvas, int imageSize) {
  for (RANSAC::Parameter* parameter : points) {
    Point2D* point = dynamic_cast<Point2D*>(parameter);
    cv::Point pt(int(point->x), int(point->y));
    cv::circle(canvas, pt, int(floor(imageSize / 200)), cv::Scalar(0, 0, 255), -1);
  }
}

void drawResult(RANSAC::Estimator<Line2DModel, Line2DModel::MODEL_SIZE>* estimator, 
    cv::Mat& canvas, int imageSize) {
  std::vector<RANSAC::Parameter*> bestInliers = estimator->getBestInliers();
  if (bestInliers.size() == 0) {
    return;
  }

  for (RANSAC::Parameter* inlier : bestInliers) {
    Point2D* point = dynamic_cast<Point2D*>(inlier);
    cv::Point pt((int)floor(point->x), (int)floor(point->y));
    cv::circle(canvas, pt, int(floor(imageSize / 200)), cv::Scalar(0, 255, 0), -1);
  }

  Line2DModel* bestLine = dynamic_cast<Line2DModel*>(estimator->getBestModel());
  if (bestLine) {
    Point2D* point1 = dynamic_cast<Point2D*>(bestLine->getModelParameters()[0]);
    Point2D* point2 = dynamic_cast<Point2D*>(bestLine->getModelParameters()[1]);
    if (point1 && point2) {
      cv::Point pt1(int(point1->x), int(point1->y));
      cv::Point pt2(int(point2->x), int(point2->y));
      drawFullLine(canvas, pt1, pt2, cv::Scalar(0, 0, 0), 2);
    }
  }
}

void showImage(const cv::Mat& canvas) {
  cv::imshow("RANSAC example - line fit", canvas);
  cv::waitKey();
}

// ---------------------------------------------------------------
// main: example of linear fit using RANSAC
// (note: to run this main, comment main in homography_model)
/*
int main(int argc, char* argv[]) {
  int dataSize = 500;
  int imageSize = 500;
  std::vector<RANSAC::Parameter*> candidatePoints = generateData(dataSize, imageSize);

  RANSAC::Estimator<Line2DModel, Line2DModel::MODEL_SIZE> estimator;
  estimator.initialize(20, 100);
  estimator.estimate(candidatePoints);
  std::cout << "RANSAC execution time: " << estimator.getExecutionTime() << " ms" << std::endl;

  cv::Mat canvas(imageSize, imageSize, CV_8UC3);
  canvas.setTo(255);
  drawPoints(candidatePoints, canvas, imageSize);
  drawResult(&estimator, canvas, imageSize);
  showImage(canvas);
}
*/
