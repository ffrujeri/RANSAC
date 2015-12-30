//
// Created by Felipe Vieira Frujeri on 29/12/15.
//

#ifndef RANSAC_RANSAC_H
#define RANSAC_RANSAC_H

#include <iostream>
#include <cmath>
#include <string>
#include <random>
#include <algorithm>
#include <vector>
#include <memory>

#include "Model.h"

namespace RANSAC {
    template <class T, int modelSize>
    class Ransac {
    private:
        std::vector<std::shared_ptr<Parameter> > data;

        std::vector<std::shared_ptr<T> > sampledModels;
        std::shared_ptr<T> bestModel;
        std::vector<std::shared_ptr<Parameter> > bestInliers;

        int maxIterations;
        VPFloat threshold;
        VPFloat bestModelScore;
        int bestModelIndex;

    public:
        void reset(void) {
            data.clear();
            sampledModels.clear();

            bestModelIndex = -1;
            bestModelScore = 0.0;
        };

        void initialize(VPFloat threshold, int maxIterations = 1000) {
            this->threshold = threshold;
            this->maxIterations = maxIterations;
        }

        std::shared_ptr<T> getBestModel(void) {
            return bestModel;
        }

        const std::vector<std::shared_ptr<Parameter> >& getBestInliers(void) {
            return bestInliers;
        }

        bool estimate(std::vector<std::shared_ptr<Parameter> > data) {
            if(data.size() <= modelSize) {
                std::cout << "number of data points is too small" << std::endl;
                return false;
            }

            this->data = data;
            std::uniform_int_distribution<int> uniDist(0, int(this->data.size()-1));

            std::vector<VPFloat> inliersFractionAccum(maxIterations);
            std::vector<std::vector<std::shared_ptr<Parameter> > > inliersAccum(maxIterations);

            sampledModels.resize(maxIterations);

            for (int i = 0; i < maxIterations; ++i) {
                std::vector<std::shared_ptr<Parameter> > randomSamples(modelSize);
                std::vector<std::shared_ptr<Parameter> > remainderSamples = this->data;

                std::random_device SeedDevice;
                std::mt19937 g(SeedDevice());
                std::shuffle(remainderSamples.begin(), remainderSamples.end(), g);
                std::copy(remainderSamples.begin(), remainderSamples.begin() + modelSize, randomSamples.begin());
                remainderSamples.erase(remainderSamples.begin(), remainderSamples.begin() + modelSize);

                std::shared_ptr<T> randomModel = std::make_shared<T>(randomSamples);

                std::pair<VPFloat, std::vector<std::shared_ptr<Parameter> > > evalPair = randomModel->evaluate(remainderSamples, threshold);
                inliersFractionAccum[i] = evalPair.first;
                inliersAccum[i] = evalPair.second;

                sampledModels[i] = randomModel;
            }

            for (int i = 0; i < maxIterations; ++i) {
                if (inliersFractionAccum[i] > bestModelScore) {
                    bestModelScore = inliersFractionAccum[i];
                    bestModelIndex = sampledModels.size() - 1;
                    bestModel = sampledModels[i];
                    bestInliers = inliersAccum[i];
                }
            }

            reset();

            return true;
        }
    };
}
#endif //RANSAC_RANSAC_H
