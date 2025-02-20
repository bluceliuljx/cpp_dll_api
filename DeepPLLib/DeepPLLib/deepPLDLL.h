#pragma once
#include "deepPLDLL_export.h"
#include "PLType.h"

#ifdef DeepPL_DLL_EXPORTS
#define DeepPL_DLL_API __declspec(dllexport)  // ��������
#else
#define DeepPL_DLL_API __declspec(dllimport)  // �������
#endif
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>



#ifdef __cplusplus
extern "C" {
#endif

	/************************************************************************/
	/*                            ���Խӿ�                                  */
	/************************************************************************/
	DeepPL_DLL_API void TestAPI();


	/************************************************************************/
	/*                        ��ˮ���ξ߼��                                */
	/************************************************************************/
	/** \brief init_model: ��ʼ�� �ξ߼�����ģ��
	*
	* \param        model_path          input: ģ��·��
	* \param        cfg_path            input: ����·��
	* \param        log_path            input: ��־Ŀ¼·��
	* \return       true if succeeds, otherwise false
	*
	*/
	//DeepPL_DLL_API bool init_model(std::string& model_path, std::string& cfg_path, std::string& log_path);
	DeepPL_DLL_API bool init_model(const char* model_path, const char* cfg_path, const char* log_path);


	/** \brief PLZJDetection: ��������ͼƬ
	*
	* \param        img_path            input: ͼ��·��
	* \param        pOutims				output: ��������������
	* \param        num					output: ��������������
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num);


	/** \brief FreeOutParams: �ͷ�pOutim
	* \param        pOutims            input: pOutims��ָ��
	* \param        num            input: ����
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API void FreeOutParams(OutParams*& pOutims, int num);


	/** \brief FreeOut: �ͷ���Դ
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API void FreeOut();


	// ������������
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
//��ˮ���ξߵ��ò���
///////////////////////////////////////////////////////////
/*
1����ʼ��
	DeepPL_DLL_API bool init_model(const char* model_path, const char* cfg_path, const char* log_path);
2������
	2.1����������ͼ��
		DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num);
	2.2��ÿ�η�����ɵ�������ͼ���Ҫ�ͷ�pOutims
		DeepPL_DLL_API void FreeOutParams(OutParams* pOutims, int num);
	2.3���ظ�2.1��2.2���裬�����ķ���ͼ��
3���ر���Դ��ģ�ͣ�������Ҫ���д�ģ�ͺ��ͷ�ģ�ͣ�
DeepPL_DLL_API void FreeOut();

*/


///////////////////////////////////////////////////////////
//��ˮ���ξ����˵��
//A��������; B�����Ĺ�; C����tipͷ; D��Сtipͷ
///////////////////////////////////////////////////////////
/*
label   info
0       A1������������
1       A2���޳�����
2       A3���г����֣���δ��ȷ���õ�������λ�ã���б��
3       A4���г����֣�����Ƭ���÷���
4       B1�����Ĺ�����
5       B2�������Ĺ�
6       B3�������Ĺܣ���ȱб������
7       C1����tipͷ����
8       C2���޴�tipͷ
9       D1��Сtipͷ����
10      D2����Сtipͷ
*/