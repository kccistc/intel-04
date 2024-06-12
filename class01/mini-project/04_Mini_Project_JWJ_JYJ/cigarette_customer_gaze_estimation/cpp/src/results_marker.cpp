// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cstdio>
#include <iostream>
#include <array>

#define _USE_MATH_DEFINES
#include <cmath>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include "results_marker.hpp"
#include "face_inference_results.hpp"
#include "utils.hpp"


// 전역 변수 선언
double yaw, pitch, roll;

// 각도를 라디안으로 변환하는 함수
double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

// 3x3 행렬을 나타내는 구조체
struct Matrix3x3 {
    std::array<std::array<double, 3>, 3> data;

    Matrix3x3() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                data[i][j] = 0.0;
            }
        }
    }

    std::array<double, 3>& operator[](size_t i) {
        return data[i];
    }

    const std::array<double, 3>& operator[](size_t i) const {
        return data[i];
    }
};


// 3D 벡터를 나타내는 구조체
struct Vector3 {
    std::array<double, 3> data;

    Vector3() {
        for (int i = 0; i < 3; ++i) {
            data[i] = 0.0;
        }
    }

    double& operator[](size_t i) {
        return data[i];
    }

    const double& operator[](size_t i) const {
        return data[i];
    }
};


// 행렬 곱셈 함수
Matrix3x3 matmul(const Matrix3x3& A, const Matrix3x3& B) {
    Matrix3x3 C;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


// 행렬과 벡터의 곱셈 함수
Vector3 matvecmul(const Matrix3x3& A, const Vector3& v) {
    Vector3 result;
    for (int i = 0; i < 3; ++i) {
        result[i] = 0;
        for (int j = 0; j < 3; ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}


// 머리 자세 각도로부터 회전 행렬을 계산하는 함수
Matrix3x3 getRotationMatrix(double yaw, double pitch, double roll) {
    // 각도를 라디안으로 변환
    yaw = deg2rad(yaw);
    pitch = deg2rad(pitch);
    roll = deg2rad(roll);

    // 회전 행렬
    Matrix3x3 R_yaw;
    R_yaw[0] = { cos(yaw), -sin(yaw), 0 };
    R_yaw[1] = { sin(yaw),  cos(yaw), 0 };
    R_yaw[2] = {       0,        0, 1 };

    Matrix3x3 R_pitch;
    R_pitch[0] = { cos(pitch), 0, sin(pitch) };
    R_pitch[1] = {         0, 1,        0 };
    R_pitch[2] = { -sin(pitch), 0, cos(pitch) };

    Matrix3x3 R_roll;
    R_roll[0] = { 1,        0,         0 };
    R_roll[1] = { 0, cos(roll), -sin(roll) };
    R_roll[2] = { 0, sin(roll),  cos(roll) };

    // 복합 회전 행렬
    Matrix3x3 R = matmul(R_yaw, matmul(R_pitch, R_roll));
    return R;
}

// 시선 방향 벡터를 계산하는 함수
Vector3 getGazeDirection(double horizontal, double vertical) {
    // 각도를 라디안으로 변환
    horizontal = deg2rad(horizontal);
    vertical = deg2rad(vertical);

    // 시선 방향 벡터
    Vector3 g;
    g[0] = cos(vertical) * sin(horizontal);
    g[1] = sin(vertical);
    g[2] = cos(vertical) * cos(horizontal);

    return g;
}


// 머리 자세와 시선 각도로부터 2D 평면상의 좌표를 계산하는 함수
std::pair<double, double> projectGaze(double yaw, double pitch, double roll, double horizontal, double vertical, double d = 1.0) {
    // 머리 자세 각도로부터 회전 행렬 계산
    Matrix3x3 R = getRotationMatrix(yaw, pitch, roll);

    // 머리 좌표계에서 시선 방향 벡터 계산
    Vector3 g_head = getGazeDirection(horizontal, vertical);

    // 시선 방향을 세계 좌표계로 변환
    Vector3 g_world = matvecmul(R, g_head);

    // 2D 평면에 투영 (d = 투영 평면까지의 거리)
    double x = (g_world[0] / g_world[2]) * d * 9.0;
    double y = (g_world[1] / g_world[2]) * d * 4.0;

    return std::make_pair(x, y);
}



namespace gaze_estimation {
ResultsMarker::ResultsMarker(
    bool showFaceBoundingBox, bool showHeadPoseAxes, bool showLandmarks, bool showGaze, bool showEyeState) :
        showFaceBoundingBox(showFaceBoundingBox),
        showHeadPoseAxes(showHeadPoseAxes),
        showLandmarks(showLandmarks),
        showGaze(showGaze),
        showEyeState(showEyeState)
{
}

void ResultsMarker::mark(cv::Mat& image, const FaceInferenceResults& faceInferenceResults) const {
    auto faceBoundingBox = faceInferenceResults.faceBoundingBox;
    auto faceBoundingBoxWidth = faceBoundingBox.width;
    auto faceBoundingBoxHeight = faceBoundingBox.height;
    auto scale =  0.002 * faceBoundingBoxWidth;
    cv::Point tl = faceBoundingBox.tl();

    if (showFaceBoundingBox) {
        cv::rectangle(image, faceInferenceResults.faceBoundingBox, cv::Scalar::all(255), 1);
        putHighlightedText(image,
                    cv::format("Detector confidence: %0.2f",
                    static_cast<double>(faceInferenceResults.faceDetectionConfidence)),
                    cv::Point(static_cast<int>(tl.x),
                    static_cast<int>(tl.y - 5. * faceBoundingBoxWidth / 200.)),
                    cv::FONT_HERSHEY_COMPLEX, scale, cv::Scalar(200, 10, 10), 1);
    }

    if (showHeadPoseAxes) {
        yaw = static_cast<double>(faceInferenceResults.headPoseAngles.x);
        pitch = static_cast<double>(faceInferenceResults.headPoseAngles.y);
        roll = static_cast<double>(faceInferenceResults.headPoseAngles.z);

        auto sinY = std::sin(yaw * M_PI / 180.0);
        auto sinP = std::sin(pitch * M_PI / 180.0);
        auto sinR = std::sin(roll * M_PI / 180.0);

        auto cosY = std::cos(yaw * M_PI / 180.0);
        auto cosP = std::cos(pitch * M_PI / 180.0);
        auto cosR = std::cos(roll * M_PI / 180.0);

        auto axisLength = 0.4 * faceBoundingBoxWidth;
        auto xCenter = faceBoundingBox.x + faceBoundingBoxWidth / 2;
        auto yCenter = faceBoundingBox.y + faceBoundingBoxHeight / 2;

        // OX points from face center to camera
        // OY points from face center to right
        // OZ points from face center to up

        // Rotation matrix:
        // Yaw - counterclockwise Pitch - counterclockwise Roll - clockwise
        //     [cosY -sinY 0]          [ cosP 0 sinP]       [1    0    0 ]
        //     [sinY  cosY 0]    *     [  0   1  0  ]   *   [0  cosR sinR] =
        //     [  0    0   1]          [-sinP 0 cosP]       [0 -sinR cosR]

        //   [cosY*cosP cosY*sinP*sinR-sinY*cosR cosY*sinP*cosR+sinY*sinR]
        // = [sinY*cosP cosY*cosR-sinY*sinP*sinR sinY*sinP*cosR+cosY*sinR]
        //   [  -sinP          -cosP*sinR                cosP*cosR       ]

        // Multiply third row by -1 because screen drawing axis points down
        // Drop first row to project to a screen plane

        // OY: center to right
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * (cosR * cosY - sinY * sinP * sinR)),
                           static_cast<int>(yCenter + axisLength * cosP * sinR)),
                 cv::Scalar(0, 0, 255), 2);
        // OZ: center to top
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * (cosR * sinY * sinP + cosY * sinR)),
                           static_cast<int>(yCenter - axisLength * cosP * cosR)),
                 cv::Scalar(0, 255, 0), 2);
        // OX: center to camera
        cv::line(image, cv::Point(xCenter, yCenter),
                 cv::Point(static_cast<int>(xCenter + axisLength * sinY * cosP),
                           static_cast<int>(yCenter + axisLength * sinP)),
                 cv::Scalar(255, 0, 255), 2);

        putHighlightedText(image,
            cv::format("head pose: (y=%0.0f, p=%0.0f, r=%0.0f)", std::round(yaw), std::round(pitch), std::round(roll)),
            cv::Point(static_cast<int>(faceBoundingBox.tl().x),
            static_cast<int>(faceBoundingBox.br().y + 5. * faceBoundingBoxWidth / 100.)),
            cv::FONT_HERSHEY_PLAIN, scale * 2, cv::Scalar(200, 10, 10), 1);
    }

    if (showLandmarks) {
        int lmRadius = static_cast<int>(0.01 * faceBoundingBoxWidth + 1);
        for (auto const& point : faceInferenceResults.faceLandmarks)
            cv::circle(image, point, lmRadius, cv::Scalar(0, 255, 255), -1);
    }

    if (showGaze) {
        auto gazeVector = faceInferenceResults.gazeVector;

        double arrowLength = 0.4 * faceBoundingBoxWidth;
        cv::Point2f gazeArrow;
        gazeArrow.x = gazeVector.x;
        gazeArrow.y = -gazeVector.y;

        gazeArrow = arrowLength * gazeArrow;

        // Draw eyes bounding boxes
        cv::rectangle(image, faceInferenceResults.leftEyeBoundingBox, cv::Scalar::all(255), 1);
        cv::rectangle(image, faceInferenceResults.rightEyeBoundingBox, cv::Scalar::all(255), 1);

        if (faceInferenceResults.leftEyeState)
            cv::arrowedLine(image,
                faceInferenceResults.leftEyeMidpoint,
                faceInferenceResults.leftEyeMidpoint + gazeArrow, cv::Scalar(255, 0, 0), 2);

        if (faceInferenceResults.rightEyeState)
            cv::arrowedLine(image,
                faceInferenceResults.rightEyeMidpoint,
                faceInferenceResults.rightEyeMidpoint + gazeArrow, cv::Scalar(255, 0, 0), 2);

        cv::Point2f gazeAngles;
        if (faceInferenceResults.leftEyeState && faceInferenceResults.rightEyeState) {
            gazeVectorToGazeAngles(faceInferenceResults.gazeVector, gazeAngles);

            auto eye_horizontal = static_cast<double>(std::round(gazeAngles.x));
            auto eye_vertical = static_cast<double>(std::round(gazeAngles.y));

            std::string cig_array[3][7];

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 7; ++j) {
                    cig_array[i][j] = "";  // 빈 문자열로 초기화
                }   
            }

            cig_array[0][0] = "ESSE Change 1mg";
            cig_array[0][1] = "ESSE Change 4mg";
            cig_array[0][2] = "ESSE Change Frozen";
            cig_array[0][3] = "ESSE Change Ice Fall";
            cig_array[0][4] = "ESSE Soo 0.5";
            cig_array[0][5] = "ESSE Soo 0.1";
            cig_array[0][6] = "ESSE One";

            cig_array[1][0] = "Marlboro Red";
            cig_array[1][1] = "Marlboro Midium";
            cig_array[1][2] = "Marlboro Gold";
            cig_array[1][3] = "Marlboro Silver";
            cig_array[1][4] = "Marlboro Ice Blast 1mg";
            cig_array[1][5] = "Marlboro Ice Blast 5mg";
            cig_array[1][6] = "Marlboro Vista";

            cig_array[2][0] = "Dunhill 6mg";
            cig_array[2][1] = "Dunhill 3mg";
            cig_array[2][2] = "Dunhill 1mg";
            cig_array[2][3] = "Dunhill Switch 6mg";
            cig_array[2][4] = "Dunhill Switch one";
            cig_array[2][5] = "Dunhill Fine-Cut 1mg";
            cig_array[2][6] = "Dunhill Fine-Cut 0.1mg";

            putHighlightedText(image,
                cv::format("gaze angles: (h=%0.0f, v=%0.0f)",
                eye_horizontal,
                eye_vertical),
                cv::Point(static_cast<int>(faceBoundingBox.tl().x),
                static_cast<int>(faceBoundingBox.br().y + 12. * faceBoundingBoxWidth / 100.)),
                cv::FONT_HERSHEY_PLAIN, scale * 2, cv::Scalar(200, 10, 10), 1);
            
            auto result = projectGaze(yaw,pitch,roll,eye_horizontal,eye_vertical);

            if (result.first < -8){
                result.first = 999;
            }
            else if (result.first >= -8 && result.first < -5){
                result.first = -3;
            }
            else if (result.first >= -5 && result.first < -3){
                result.first = -2;
            }
            else if (result.first >= -3 && result.first < -1){
                result.first = -1;
            }
            else if (result.first >= -1 && result.first < 1){
                result.first = 0;
            }
            else if (result.first >= 1 && result.first < 3){
                result.first = 1;
            }
            else if (result.first >= 3 && result.first < 5){
                result.first = 2;
            }
            else if (result.first >= 5 && result.first < 8){
                result.first = 3;
            }
            else if (result.first >= 8){
                result.first = 999;
            }

            if (result.second <= -2 || result.second >= 2){
                result.second = 999;
            }

            if (result.first >= 999 || result.second >= 999){
                putHighlightedText(image,
                cv::format("Cigarette Point : Out of Range"),
                cv::Point(static_cast<int>(faceBoundingBox.tl().x)-100,
                static_cast<int>(faceBoundingBox.br().y + 19. * faceBoundingBoxWidth / 100.)),
                cv::FONT_HERSHEY_PLAIN, scale * 3, cv::Scalar(10, 200, 10), 1);
                result.first = 0;
                result.second = 0;
            }
            else{

                    // 부동 소수점 값을 정수로 변환 (반올림)
                int indexFirst = static_cast<int>(std::floor(result.first)+3.0);
                int indexSecond = static_cast<int>(std::trunc(result.second)+1.0);                

                putHighlightedText(image,
                cv::format("Cigarette Point : (x=%0.0f, y=%0.0f)",
                result.first,
                result.second),
                cv::Point(static_cast<int>(faceBoundingBox.tl().x)-100,
                static_cast<int>(faceBoundingBox.br().y + 19. * faceBoundingBoxWidth / 100.)),
                cv::FONT_HERSHEY_PLAIN, scale * 3, cv::Scalar(10, 200, 10), 1);

                putHighlightedText(image,
                cv::format("Cigarette Name : (%s)",cig_array[indexSecond][indexFirst].c_str()),
                cv::Point(static_cast<int>(faceBoundingBox.tl().x)-100,
                static_cast<int>(faceBoundingBox.br().y + 26. * faceBoundingBoxWidth / 100.)),
                cv::FONT_HERSHEY_PLAIN, scale * 3, cv::Scalar(10, 200, 10), 1);

            }            
        }
    }
    if (showEyeState) {
        if (faceInferenceResults.leftEyeState)
            cv::rectangle(image, faceInferenceResults.leftEyeBoundingBox, cv::Scalar(0, 255, 0), 1);
        else
            cv::rectangle(image, faceInferenceResults.leftEyeBoundingBox, cv::Scalar(0, 0, 255), 1);

        if (faceInferenceResults.rightEyeState)
            cv::rectangle(image, faceInferenceResults.rightEyeBoundingBox, cv::Scalar(0, 255, 0), 1);
        else
            cv::rectangle(image, faceInferenceResults.rightEyeBoundingBox, cv::Scalar(0, 0, 255), 1);
    }
}

void ResultsMarker::toggle(int key) {
    switch (std::toupper(key)) {
        case 'L':
            showLandmarks = !showLandmarks;
            break;
        case 'O':
            showHeadPoseAxes = !showHeadPoseAxes;
            break;
        case 'G':
            showGaze = !showGaze;
            break;
        case 'B':
            showFaceBoundingBox = !showFaceBoundingBox;
            break;
        case 'A':
            showFaceBoundingBox = true;
            showHeadPoseAxes = true;
            showLandmarks = true;
            showGaze = true;
            break;
        case 'N':
            showFaceBoundingBox = false;
            showHeadPoseAxes = false;
            showLandmarks = false;
            showGaze = false;
            break;
        case 'E':
            showEyeState = !showEyeState;
            break;
    }
}
}  // namespace gaze_estimation
