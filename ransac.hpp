#include <vector>
#include <array>

// generic RANSAC
// usage:
// - to implement custom RANSAC, extend Parameter and Model<MODEL_SIZE> classes 
//   and implement necessary methods
// - use class Estimator to apply custom RANSAC:
//      RANSAC::Estimator<CustomModel, MODEL_SIZE> estimator;
//      estimator.initialize(20, 100);
//      estimator.estimate(candidatePoints);

namespace RANSAC {
  class Parameter {
  public:
    // virtual function necessary for dynamic cast
    virtual ~Parameter() {}
  };

  template <int modelSize> class Model {
  public:
    virtual void initialize(const std::vector<Parameter*>& inputParameters) = 0;
    virtual std::pair<double, std::vector<Parameter*> > evaluate(
        std::vector<Parameter*> inputParameters, double threshold) = 0;

    virtual std::array<Parameter*, modelSize> getModelParameters() {
      return minModelParameters;
    };

  protected:
    std::array<Parameter*, modelSize> minModelParameters;

    virtual double computeDistance(Parameter* parameter) = 0;
  };
  
  template <class ModelImpl, int modelSize> class Estimator {
  public:
    void initialize(double threshold, int maxIterations = 1000) {
      this->threshold = threshold;
      this->maxIterations = maxIterations;
    }

    bool estimate(const std::vector<Parameter*>& data) {
      if (data.size() <= modelSize) {
        std::cout << "data insuficient to compute best model" << std::endl;
        executionTime = -1;
        return false;
      }

      int64 startTime = cv::getTickCount();
      this->data = data;
      std::uniform_int_distribution<int> uniDist(0, int(this->data.size() - 1));
      
      for (int i = 0; i < maxIterations; i++) {
        std::vector<Parameter*> randomSamples(modelSize);
        std::vector<Parameter*> remainderSamples = this->data;
        std::random_device SeedDevice;
        std::mt19937 g(SeedDevice());
        std::shuffle(remainderSamples.begin(), remainderSamples.end(), g);
        std::copy(
            remainderSamples.begin(), remainderSamples.begin() + modelSize, randomSamples.begin());
        remainderSamples.erase(remainderSamples.begin(), remainderSamples.begin() + modelSize);

        ModelImpl randomModel(randomSamples);
        std::pair<double, std::vector<Parameter*> > evalPair = 
            randomModel.evaluate(remainderSamples, threshold);
        double score = evalPair.first;

        if (score > bestModelScore) {
          bestModelScore = score;
          bestModel = new ModelImpl(randomSamples);
          bestInliers = evalPair.second;
        }
      }

      reset();
      executionTime = double(cv::getTickCount() - startTime) / double(cv::getTickFrequency()) * 1000;
      return true;
    }

    void reset() {
      data.clear();
      bestModelScore = 0.0;
    };

    ModelImpl* getBestModel() {
      return bestModel;
    }

    const std::vector<Parameter*>& getBestInliers() {
      return bestInliers;
    }

    double getExecutionTime() {
      return executionTime;
    }

  private:
    int maxIterations;
    double threshold;
    std::vector<Parameter*> data;

    ModelImpl* bestModel;
    std::vector<Parameter*> bestInliers;
    double bestModelScore;
    double executionTime;
  };
}
