//
// Created by Felipe Vieira Frujeri on 29/12/15.
//

#ifndef RANSAC_LINEMODEL_H
#define RANSAC_LINEMODEL_H

#include <cmath>
#include <memory>

#include "include/Model.h"

typedef std::array<RANSAC::VPFloat, 2> Vector2VP;

class Point2D : public RANSAC::Parameter {
public:
    Vector2VP point2D;

    Point2D(RANSAC::VPFloat x, RANSAC::VPFloat y) {
        point2D[0] = x;
        point2D[1] = y;
    };

};

class Line2DModel : public RANSAC::Model<2> {
protected:
    RANSAC::VPFloat a, b, c;
    RANSAC::VPFloat distDenominator;

    RANSAC::VPFloat  m;
    RANSAC::VPFloat  d;

    virtual RANSAC::VPFloat computeDistance(std::shared_ptr<RANSAC::Parameter> param) override {
        auto extPoint2D = std::dynamic_pointer_cast<Point2D>(param);

        if (extPoint2D == nullptr) {
            throw std::runtime_error("Line2DModel - Passed parameters are not of type type Point2D");
        }

        RANSAC::VPFloat numer = fabs(a * extPoint2D->point2D[0] + b * extPoint2D->point2D[1] + c);
        RANSAC::VPFloat dist = numer / distDenominator;

        return dist;
    };
public:
    Line2DModel(std::vector<std::shared_ptr<RANSAC::Parameter> > inputParams) {
        initialize(inputParams);
    }

    virtual void initialize(std::vector<std::shared_ptr<RANSAC::Parameter> > inputParams) override {
        if (inputParams.size() != 2) {
            throw std::runtime_error("Line2DModel - Number of input parameters does not match minimum number required for this model.");
        }
        auto point1 = std::dynamic_pointer_cast<Point2D>(inputParams[0]);
        auto point2 = std::dynamic_pointer_cast<Point2D>(inputParams[1]);

        if (point1 == nullptr || point2 == nullptr) {
            throw std::runtime_error("Line2DModel - InputParams type mismatch. It is not a Point2D.");
        }

        std::copy(inputParams.begin(), inputParams.end(), minModelParams.begin());

        m = (point2->point2D[1] - point1->point2D[1]) / (point2->point2D[0] - point1->point2D[0]);
        d = point1->point2D[1] - m * point1->point2D[0];

        a = m;
        b = -1.0;
        c = d;

        distDenominator = sqrt(a*a + b*b);
    }

    virtual std::pair<RANSAC::VPFloat, std::vector<std::shared_ptr<RANSAC::Parameter> > > evaluate(
            std::vector<std::shared_ptr<RANSAC::Parameter> > evaluateParams, RANSAC::VPFloat threshold) override{

        std::vector<std::shared_ptr<RANSAC::Parameter> > inliers;
        int nTotalParams = int(evaluateParams.size());
        int nInliers = 0;

        for(auto& param : evaluateParams) {
            if (computeDistance(param) < threshold) {
                inliers.push_back(param);
                nInliers++;
            }
        }

        RANSAC::VPFloat inliersFraction = RANSAC::VPFloat(nInliers) / RANSAC::VPFloat(nTotalParams);

        return std::make_pair(inliersFraction, inliers);
    };

};

#endif //RANSAC_LINEMODEL_H
