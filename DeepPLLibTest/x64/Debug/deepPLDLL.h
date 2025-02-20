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
	* \param        log_path            input: ��־·��
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API bool init_model(std::string& model_path, std::string& cfg_path, std::string& log_path);


	/** \brief PLZJDetection: ��������ͼƬ
	*
	* \param        img_path            input: ͼ��·��
	* \param        pOutims				output: ��������������
	* \param        num					output: ��������������
	* \return       true if succeeds, otherwise false
	*
	*/
	DeepPL_DLL_API bool PLZJDetection(std::string& img_path, std::vector<OutParams>& pOutims, int& num);


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