#include <string.h>
#include <jni.h>
#include <math.h>
#include <iostream>
#include <vector>
#include "nms.h"
#include <android/log.h>

using namespace std;

float sigmoid(float f) {
    return (float)(1.0f / (1.0f + exp(-f)));
}

float revsigmoid(float f){
    const float eps = 1e-8;
    return -1.0f * (float)log((1.0f / (f + eps)) - 1.0f);
}
#define CLASS_NUM 80
#define max_wh 4096
#define LOG_TAG "TFLITE_RUNNER"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)


void detector(
        vector<bbox>* bbox_candidates,
        JNIEnv *env,
        jobjectArray input,
        const int num_boxes,
        const float conf_thresh) {

    if (input == NULL) {
        LOGE("Input array is NULL");
        return;
    }

    float revsigmoid_conf = revsigmoid(conf_thresh);
    revsigmoid_conf = conf_thresh;
    for (int bi = 0; bi < 1; bi++) {
        jobjectArray batchArray = (jobjectArray) env->GetObjectArrayElement(input, bi);
        for (int i = 0; i < num_boxes; i++) {
            jobjectArray objectArray = (jobjectArray) env->GetObjectArrayElement(batchArray, i);
            auto elmptr = env->GetFloatArrayElements((jfloatArray)objectArray, nullptr);

            // 각 그리드 셀의 데이터를 처리
            auto elmptr_ch = elmptr;
            // 객체 존재 확률 확인
            float score = elmptr_ch[4];

            if (score >= revsigmoid_conf) {
                // 클래스 확률 계산 및 가장 높은 클래스 찾기
                float max_class_conf = elmptr_ch[5];
                int max_class_idx = 0;
                for (int class_idx = 1; class_idx < CLASS_NUM; class_idx++) {
                    float class_conf = elmptr_ch[class_idx + 5];
                    if (class_conf > max_class_conf) {
                        max_class_conf = class_conf;
                        max_class_idx = class_idx;
                    }
                }
                LOGD("[gunhot]max_class_idx: %d", max_class_idx);

                // 바운딩 박스 신뢰도 계산
                float bbox_conf = sigmoid(max_class_conf) * sigmoid(score);
                float cx = elmptr_ch[0];
                float cy = elmptr_ch[1];
                float w = elmptr_ch[2];
                float h = elmptr_ch[3];
                float x1 = cx - w / 2.0f;
                float y1 = cy - h / 2.0f;
                float x2 = cx + w / 2.0f;
                float y2 = cy + h / 2.0f;
                LOGD("[gunhot](x1, y1) = (%f, %f) | (x2, y2) = (%f, %f)", x1, y1, x2, y2);
                // 바운딩 박스 객체 생성 및 후보 리스트에 추가
                bbox box = bbox(x1, y1, x2, y2, bbox_conf, max_class_idx);
                bbox_candidates->push_back(box);
            }

            // JNI 메모리 해제
            env->ReleaseFloatArrayElements((jfloatArray)objectArray, elmptr, 0);
            env->DeleteLocalRef(objectArray);
        }

        env->DeleteLocalRef(batchArray);
    }
    env->DeleteLocalRef(input);
}



extern "C" jobjectArray Java_com_example_tflite_1yolov5_1test_TfliteRunner_postprocess(
        JNIEnv *env,
        jobject /* this */,
        jobjectArray input,
        jint input_size,
        jfloat conf_thresh,
        jfloat iou_thresh) {

    vector<bbox> bbox_candidates;

    LOGD("bbox start");
    // YOLOX_nano에서 단일 출력 텐서에 대해 탐지를 수행
    detector(&bbox_candidates, env, input, 2100, conf_thresh);
    LOGD("bbox well maded");
    // 비최대 억제 과정
    vector<bbox> nms_results = nms(bbox_candidates, iou_thresh);

    jobjectArray objArray;
    jclass floatArray = env->FindClass("[F");
    if (floatArray == NULL) return NULL;
    int size = nms_results.size();
    objArray = env->NewObjectArray(size, floatArray, NULL);
    if (objArray == NULL) return NULL;

    for (int i = 0; i < nms_results.size(); i++) {
        int class_idx = nms_results[i].class_idx;
        float x1 = nms_results[i].x1;
        float y1 = nms_results[i].y1;
        float x2 = nms_results[i].x2;
        float y2 = nms_results[i].y2;
        float conf = nms_results[i].conf;
        float boxres[6] = {x1, y1, x2, y2, conf, (float)class_idx};
        jfloatArray iarr = env->NewFloatArray((jsize)6);
        if (iarr == NULL) return NULL;
        env->SetFloatArrayRegion(iarr, 0, 6, boxres);
        env->SetObjectArrayElement(objArray, i, iarr);
        env->DeleteLocalRef(iarr);
    }
    return objArray;
}
