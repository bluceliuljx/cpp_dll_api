#pragma once
#include "deepPLDLL_export.h"
#include "PLType.h"

#ifdef DeepPL_DLL_EXPORTS
#define DeepPL_DLL_API __declspec(dllexport)  // 导出符号
#else
#define DeepPL_DLL_API __declspec(dllimport)  // 导入符号
#endif
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>



#ifdef __cplusplus
extern "C" {
#endif

	/************************************************************************/
	/*                            测试接口                                  */
	/************************************************************************/
	DeepPL_DLL_API void TestAPI();


	/************************************************************************/
	/*                        流水线治具检测                                */
	/************************************************************************/
	/** \brief init_model: 初始化 治具检测分析模型
	*
	* \param        model_path          input: 模型路径
	* \param        cfg_path            input: 配置路径
	* \param        log_path            input: 日志目录路径
	* \return       true if succeeds, otherwise false
	*
	*/
	//DeepPL_DLL_API bool init_model(std::string& model_path, std::string& cfg_path, std::string& log_path);
	DeepPL_DLL_API bool init_model(const char* model_path, const char* cfg_path, const char* log_path);


	/** \brief PLZJDetection: 分析单张图片
	*
	* \param        img_path            input: 图像路径
	* \param        pOutims				output: 输出检测结果的数组
	* \param        num					output: 输出检测结果的数量
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num);


	/** \brief FreeOutParams: 释放pOutim
	* \param        pOutims            input: pOutims的指针
	* \param        num            input: 长度
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API void FreeOutParams(OutParams*& pOutims, int num);


	/** \brief FreeOut: 释放资源
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API void FreeOut();


	// 其他辅助函数
	DeepPL_DLL_API bool preprocess(std::string& img_path,
		int input_w,
		int input_h,
		float& x_factor,
		float& y_factor,
		cv::Mat& image_origin,
		cv::Mat& o_image);


#ifdef __cplusplus
}
#endif


///////////////////////////////////////////////////////////
//流水线治具调用步骤
///////////////////////////////////////////////////////////
/*
1、初始化
	DeepPL_DLL_API bool init_model(const char* model_path, const char* cfg_path, const char* log_path);
2、分析
	2.1、分析单张图像
		DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num);
	2.2、每次分析完成单纯单张图像后，要释放pOutims
		DeepPL_DLL_API void FreeOutParams(OutParams* pOutims, int num);
	2.3、重复2.1、2.2步骤，持续的分析图像
3、关闭资源（模型；不在需要运行此模型后，释放模型）
DeepPL_DLL_API void FreeOut();

*/


///////////////////////////////////////////////////////////
//流水线治具类别说明
//A、沉降仓; B、离心管; C、大tip头; D、小tip头
///////////////////////////////////////////////////////////
/*
label   info
0       A1：沉降仓正常
1       A2：无沉降仓
2       A3：有沉降仓，但未正确放置到沉降仓位置（倾斜）
3       A4：有沉降仓，但玻片放置反了
4       B1：离心管正常
5       B2：无离心管
6       B3：有离心管，但缺斜口滤网
7       C1：大tip头正常
8       C2：无大tip头
9       D1：小tip头正常
10      D2：无小tip头
*/