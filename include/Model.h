//
// Created by Felipe Vieira Frujeri on 29/12/15.
//

#ifndef RANSAC_MODEL_H
#define RANSAC_MODEL_H

#include <vector>
#include <array>
#include <memory>

namespace RANSAC {
    typedef double VPFloat;

    class Parameter {
    public:
        virtual ~Parameter() {};
    };

    template <int modelSize>
    class Model {
    protected:
        std::array<std::shared_ptr<Parameter>, modelSize> minModelParams;
        virtual VPFloat computeDistance(std::shared_ptr<Parameter> param) = 0;

    public:
        virtual void initialize(std::vector<std::shared_ptr<Parameter> > inputParams) = 0;
        virtual std::pair<VPFloat, std::vector<std::shared_ptr<Parameter> > > evaluate(
                std::vector<std::shared_ptr<Parameter> > evaluateParams,
                VPFloat threshold) = 0;

        virtual std::array<std::shared_ptr<Parameter>, modelSize> getModelParams(void) {
            return minModelParams;
        };
    };
}

#endif //RANSAC_MODEL_H
