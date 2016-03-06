#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <random>
#include <cmath>

#include "ransac.hpp"

// ---------------------------------------------------------------
// Homography model implementation
// - Point2DPair: Pairs of points representing a match between two images
// - HomographyModel: model for homography, in which matrix H is computed
//   from 4 matches of points and the distance from a set of matches to H
//   can be calculated
class Point2DPair : public RANSAC::Parameter {
public:
  const double x1, y1, x2, y2;
  Point2DPair(double x1, double y1, double x2, double y2) : x1(x1), y1(y1), x2(x2), y2(y2) {};
};

class HomographyModel : public RANSAC::Model<4> {
public:
  static const int MODEL_SIZE = 4;

  HomographyModel(const std::vector<RANSAC::Parameter*>& inputParameters) {
    initialize(inputParameters);
  }

  virtual void initialize(const std::vector<RANSAC::Parameter*>& inputParameters) override {
    if (inputParameters.size() != MODEL_SIZE) {
      throw std::runtime_error("HomographyModel - wrong size for input parameters");
    }

    std::copy(inputParameters.begin(), inputParameters.end(), minModelParameters.begin());

    cv::Mat A(A_ROWS, A_COLS, CV_64F);
    cv::Mat b(A_ROWS, 1, CV_64F);
    computeAb(A, b, inputParameters);
    computeH(A, b);
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

  cv::Mat* getHomography() {
    return H;
  }

protected:
  virtual double computeDistance(RANSAC::Parameter* parameter) override {
    Point2DPair* match = dynamic_cast<Point2DPair*>(parameter);

    double w = 1. / (H->at<double>(2, 0) * match->x1 + H->at<double>(2, 1) * match->y1 + 1.);
    double dx = (H->at<double>(0, 0) * match->x1 + H->at<double>(0, 1) * match->y1 + H->at<double>(0, 2))*w - match->x2;
    double dy = (H->at<double>(1, 0) * match->x1 + H->at<double>(1, 1) * match->y1 + H->at<double>(1, 2))*w - match->y2;

    return sqrt(dx*dx + dy*dy);
  };

private:
  static const int H_ROWS = 3;
  static const int H_COLS = 3;
  static const int H_DIM = 9;
  static const int A_ROWS = 2 * MODEL_SIZE;
  static const int A_COLS = H_DIM - 1;

  cv::Mat* H;

  void computeAb(cv::Mat& A, cv::Mat& b, const std::vector<RANSAC::Parameter*>& inputParameters) {
    for (int i = 0; i < MODEL_SIZE; i++) {
      Point2DPair* match = dynamic_cast<Point2DPair*>(inputParameters[i]);
      if (match == NULL) {
        throw std::runtime_error("HomographyModel - parameter mismatch (expected Point2DPair)");
      }

      // line 2*i
      int index = 2 * i;
      A.at<double>(index, 0) = match->x1;
      A.at<double>(index, 1) = match->y1;
      A.at<double>(index, 2) = 1.;
      A.at<double>(index, 3) = 0.;
      A.at<double>(index, 4) = 0.;
      A.at<double>(index, 5) = 0.;
      A.at<double>(index, 6) = -match->x1 * match->x2;
      A.at<double>(index, 7) = -match->x2 * match->y1;

      b.at<double>(index, 0) = match->x2;

      // line 2*i + 1
      index++;
      A.at<double>(index, 0) = 0.;
      A.at<double>(index, 1) = 0.;
      A.at<double>(index, 2) = 0.;
      A.at<double>(index, 3) = match->x1;
      A.at<double>(index, 4) = match->y1;
      A.at<double>(index, 5) = 1.;
      A.at<double>(index, 6) = -match->x1 * match->y2;
      A.at<double>(index, 7) = -match->y1 * match->y2;

      b.at<double>(index, 0) = match->y2;
    }
  }

  void computeH(const cv::Mat& A, const cv::Mat& b) {
    H = new cv::Mat(H_ROWS, H_COLS, CV_64F);
    cv::Mat auxH(H_DIM - 1, 1, CV_64F);
    cv::solve(A, b, auxH);
    for (int i = 0; i < H_ROWS; i++)
      for (int j = 0; j < H_COLS; j++)
        if (i != 2 || j != 2)
          H->at<double>(i, j) = auxH.at<double>(i * H_COLS + j, 0);
    H->at<double>(H_ROWS - 1, H_COLS - 1) = 1;
  }
};

// ---------------------------------------------------------------
// functions used main:
// > compute keypoints, compute matches, compute homgoraphy
// > draw matches
// > find horizontal homography for 2 images
// > build horizontal panorama from a vector of images
// > stitch images
enum KeypointDetectorType {AKAZE, ORB};

void computeKeyPoints(const cv::Mat& image, std::vector<cv::KeyPoint>& kpts, cv::Mat& descriptor, KeypointDetectorType detectorType) {
  if (detectorType == AKAZE) {
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(image, cv::noArray(), kpts, descriptor);
  } else if (detectorType == ORB) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    orb->detectAndCompute(image, cv::noArray(), kpts, descriptor);
  }
}

void matchPoints(const cv::Mat& descriptor1, const cv::Mat& descriptor2, std::vector<cv::DMatch>& matches){
  cv::BFMatcher M(cv::NORM_L2);
  M.match(descriptor1, descriptor2, matches);
}

void drawMatches(
    const cv::Mat& image1, const std::vector<cv::KeyPoint>& keyPoints1,
    const cv::Mat& image2, const std::vector<cv::KeyPoint>& keyPoints2,
    std::vector<cv::DMatch>& matches) {
  cv::Mat J;
  drawMatches(image1, keyPoints1, image2, keyPoints2, matches, J);
  resize(J, J, cv::Size(), .5, .5);
  imshow("Matches", J);
  cv::waitKey(0);
}

// This method considers image1 (a cummulative panorama) should be stitched to image2.
// In this case, it only considers a portion of image1 (from right to left) to find 
// its matches with image2, namely a fraction of image2 dimensions.
void findHomography(
    const cv::Mat& image1, const cv::Mat& image2,
    KeypointDetectorType detectorType, double threshold, int numIterations, cv::Mat& H, 
    double horizontalFraction) {
  std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
  cv::Mat descriptor1, descriptor2;
  std::vector<cv::DMatch> matches;

  int dx = image1.cols - image2.cols * horizontalFraction;
  cv::Rect roi(dx, 0, image2.cols * horizontalFraction, image1.rows);
  computeKeyPoints(image1(roi), keyPoints1, descriptor1, detectorType);
  // computeKeyPoints(image1, keyPoints1, descriptor1, detectorType);
  computeKeyPoints(image2, keyPoints2, descriptor2, detectorType);
  std::cout << "> computed key points" << std::endl;

  matchPoints(descriptor1, descriptor2, matches);
  std::cout << "> computed matches" << std::endl;
  // drawMatches(image1, keyPoints1, image2, keyPoints2, matches);
  
  std::vector<RANSAC::Parameter*> candidateMatches;
  for (size_t i = 0; i < matches.size(); i++) {
    Point2DPair* match = new Point2DPair(
      keyPoints1[matches[i].queryIdx].pt.x + dx, keyPoints1[matches[i].queryIdx].pt.y,
      keyPoints2[matches[i].trainIdx].pt.x, keyPoints2[matches[i].trainIdx].pt.y);
    candidateMatches.push_back(match);
  }

  RANSAC::Estimator<HomographyModel, HomographyModel::MODEL_SIZE> estimator;
  std::vector<RANSAC::Parameter*> inliers;
  HomographyModel* bestHomography;
  estimator.initialize(threshold, numIterations);
  estimator.estimate(candidateMatches);

  inliers = estimator.getBestInliers();
  bestHomography = dynamic_cast<HomographyModel*>(estimator.getBestModel());
  if (!bestHomography) {
    std::cout << "> could not compute homography" << std::endl;
  } else {
    H = (*(bestHomography->getHomography()));

    std::cout << "> computed homography" << std::endl;
    std::cout << "\t-> RANSAC execution time: " << estimator.getExecutionTime() << " ms" << std::endl;
    // std::cout << "\t->HOMOGRAPHY = " << std::endl << H << std::endl;
    std::cout << "\t-> " << inliers.size() << " inliers out of " << matches.size() << " matches" << std::endl;
  }
}

// Before stitching image1 to image2, the right offsets should be computed in
// order to consider the ROI (region of interest) and stitch the images correctly
cv::Mat stitch(const cv::Mat &image1, const cv::Mat &image2, const cv::Mat &H) {
  // coordinates of the 4 corners of the image
  std::vector<cv::Point2f> corners(4);
  corners[0] = cv::Point2f(0, 0);
  corners[1] = cv::Point2f(0, image2.rows);
  corners[2] = cv::Point2f(image2.cols, 0);
  corners[3] = cv::Point2f(image2.cols, image2.rows);

  std::vector<cv::Point2f> cornersTransform(4);
  cv::perspectiveTransform(corners, cornersTransform, H);

  double offsetX = 0.0;
  double offsetY = 0.0;

  // get max offset outside of the image
  for (size_t i = 0; i < 4; i++) {
    // std::cout << "cornersTransform[" << i << "] =" << cornersTransform[i] << std::endl;
    if (cornersTransform[i].x < offsetX) {
      offsetX = cornersTransform[i].x;
    }

    if (cornersTransform[i].y < offsetY) {
      offsetY = cornersTransform[i].y;
    }
  }

  offsetX = -offsetX;
  offsetY = -offsetY;
  // std::cout << "offsetX = " << offsetX << " ; offsetY = " << offsetY << std::endl;

  // get max width and height for the new size of the panorama
  double maxX = std::max((double)image1.cols + offsetX, (double)std::max(cornersTransform[2].x, cornersTransform[3].x) + offsetX);
  double maxY = std::max((double)image1.rows + offsetY, (double)std::max(cornersTransform[1].y, cornersTransform[3].y) + offsetY);
  // std::cout << "maxX = " << maxX << " ; maxY = " << maxY << std::endl;

  cv::Size size_warp(maxX, maxY);
  cv::Mat panorama(size_warp, CV_8UC3);

  // create the transformation matrix to be able to have all the pixels
  cv::Mat H2 = cv::Mat::eye(3, 3, CV_64F);
  H2.at<double>(0, 2) = offsetX;
  H2.at<double>(1, 2) = offsetY;

  cv::warpPerspective(image2, panorama, H2*H, size_warp);

  // ROI for image1
  cv::Rect image1_rect(offsetX, offsetY, image1.cols, image1.rows);
  
  // copy image1 in the panorama using the ROI
  cv::Mat half = cv::Mat(panorama, image1_rect);
  image1.copyTo(half);

  // create the new mask matrix for the panorama
  cv::Mat mask = cv::Mat::ones(image2.size(), CV_8U) * 255;
  cv::warpPerspective(mask, mask, H2*H, size_warp);
  cv::rectangle(mask, image1_rect, cv::Scalar(255), -1);

  return panorama;
}

void buildPanorama(
    std::vector<cv::Mat>& images, cv::Mat& result,
    KeypointDetectorType detectorType, double threshold, int numIterations, int startIndex, 
    double horizontalFraction) {
  if (images.size() < 2) {
    return;
  }

  result = images[0];
  for (size_t i = 1; i < images.size(); i++) {
    std::string index = "[" + std::to_string(startIndex) + ".." + std::to_string(startIndex + i) + "]";
    std::cout << "> creating panorama of images" << index << std::endl;
    cv::Mat H;
    findHomography(result, images[i], detectorType, threshold, numIterations, H, horizontalFraction);
    result = stitch(images[i], result, H);
    std::cout << "> stitched panorama of images " << index << std::endl;
    cv::imwrite("panorama " + index + ".png", result);
    std::cout << "> saved to file \"panorama " << index << ".png\"" << std::endl << std::endl;
  }
}

// ---------------------------------------------------------------
// main: example of finding homography and creating a panorama
// for a set of images using RANSAC
int main(int argc, char* argv[]) {
  cv::Mat result;
  KeypointDetectorType detectorType = AKAZE;
  double threshold = 3;
  int numIterations = 10000;

  // panorama 1
  int numImages = 3;
  int startIndex = 6;
  std::string prefix = "../images/pano1/image000";
  std::string suffix = ".JPG";
  std::vector<cv::Mat> images1(numImages);
  for (int i = 0; i < numImages; i++) {
    std::string path = prefix + std::to_string(startIndex + i) + suffix;
    images1[i] = cv::imread(path);
  }
  buildPanorama(images1, result, detectorType, threshold, numIterations, startIndex, 1);

  // panorama 2
  // detectorType = AKAZE;
  numIterations = 10000;
  numImages = 6;
  startIndex = 29;
  prefix = "../images/pano2/IMG_00";
  std::vector<cv::Mat> images2(numImages);
  for (int i = 0; i < numImages; i++) {
    std::string path = prefix + std::to_string(startIndex + i) + suffix;
    images2[i] = cv::imread(path);
  }
  buildPanorama(images2, result, detectorType, threshold, numIterations, startIndex, 3./5.);

  numImages = 6;
  startIndex = 44;
  for (int i = 0; i < numImages; i++) {
    std::string path = prefix + std::to_string(startIndex + i) + suffix;
    images2[i] = cv::imread(path);
  }
  buildPanorama(images2, result, detectorType, threshold, numIterations, startIndex, 3. / 5.);
  
  return 0;
}
