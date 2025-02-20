#include "deepPLDLL.h"
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <locale>
#include <codecvt>
#include <sstream>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;

///////////////////////////////////////////////////////////
//ģ��������������
//1����ʼ��ģ��
//2����ȡͼ��ͼ��Ԥ�������
//3��ģ������
//4���������NMS��
//5���ͷ���Դ
///////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////
// ʵ�� TestAPI ����
///////////////////////////////////////////////////////////
void TestAPI() {
	std::cout << "Test API start!" << std::endl;
};


///////////////////////////////////////////////////////////
// ����ȫ�ֱ���
///////////////////////////////////////////////////////////
//std::string onnxpath = "E:\\work_program\\lsx_ort_cpp_demo\\zj_ort_cpp_dll\\deepPL\\DeepPLLibTest\\x64\\Debug\\data\\zhiju.onnx";
//std::string config_path = "E:\\work_program\\lsx_ort_cpp_demo\\zj_ort_cpp_dll\\deepPL\\DeepPLLibTest\\x64\\Debug\\data\\deep_zhiju.cfg";
std::string onnxpath = "data\\zhiju.onnx";
std::string config_path = "data\\deep_zhiju.cfg";
Ort::Session session{ nullptr };
Ort::Value input_tensor{ nullptr };
float x_factor;
float y_factor;
int input_w = 1280;
int input_h = 1280;
int using_nms;
double conf_threshold;
double iou_threshold;
int device;
//std::vector<std::string> classNames;
std::string log_path_all;
std::string message;



// ���ý�����
class ConfigParser {
public:
	// ���������ļ�
	bool load(const std::string& filepath) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			std::cerr << "Unable to open config file: " << filepath << std::endl;
			return false;
		}

		std::string line;
		while (std::getline(file, line)) {
			line = trim(line);

			// ����ע���кͿ���
			if (line.empty() || line[0] == '#') {
				continue;
			}

			// ���� '=' ���ŷָ�����ֵ
			size_t pos = line.find('=');
			if (pos == std::string::npos) {
				std::cerr << "Invalid config format: " << line << std::endl;
				continue;
			}

			std::string key = trim(line.substr(0, pos));
			std::string value = trim(line.substr(pos + 1));

			if (!key.empty() && !value.empty()) {
				config_data[key] = value;
			}
		}
		return true;
	}

	// ��ȡ�ַ���ֵ
	std::string getString(const std::string& key, const std::string& default_value = "") const {
		auto it = config_data.find(key);
		if (it != config_data.end()) {
			return it->second;
		}
		return default_value;
	}

	// ��ȡ����ֵ
	int getInt(const std::string& key, int default_value = 0) const {
		auto it = config_data.find(key);
		if (it != config_data.end()) {
			try {
				return std::stoi(it->second);
			}
			catch (...) {
				std::cerr << "Invalid integer value for key: " << key << std::endl;
			}
		}
		return default_value;
	}

	// ��ȡ����ֵ
	double getDouble(const std::string& key, double default_value = 0.0) const {
		auto it = config_data.find(key);
		if (it != config_data.end()) {
			try {
				return std::stod(it->second);
			}
			catch (...) {
				std::cerr << "Invalid double value for key: " << key << std::endl;
			}
		}
		return default_value;
	}

	// ��ȡ����ֵ
	bool getBool(const std::string& key, bool default_value = false) const {
		auto it = config_data.find(key);
		if (it != config_data.end()) {
			std::string value = it->second;
			return value == "true" || value == "1";
		}
		return default_value;
	}

private:
	std::unordered_map<std::string, std::string> config_data;

	// ȥ���ַ���ǰ��ո�
	static std::string trim(const std::string& str) {
		size_t start = str.find_first_not_of(" \t");
		size_t end = str.find_last_not_of(" \t");
		return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
	}
};


//��ȡ��ǰ����
std::string getCurrentDate() {
	// ��ȡ��ǰʱ��
	auto now = std::chrono::system_clock::now();

	// ת��Ϊϵͳʱ�� (time_t)
	std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

	// ʹ�� std::tm �ṹ����������ǰʱ��
	std::tm localTime;
#ifdef _WIN32
	localtime_s(&localTime, &currentTime); // �� Windows ��ʹ�� localtime_s
#else
	localtime_r(&currentTime, &localTime); // �� Unix ϵͳ��ʹ�� localtime_r
#endif

	// ʹ�� stringstream ����ʽ��ʱ��
	std::stringstream ss;
	ss << std::put_time(&localTime, "%Y%m%d");  // ʹ�������ո�ʽ��yyyyMMdd

	return ss.str();
}

//�����ļ���
bool create_dir(std::string file_path) {

	fs::path dir_path = file_path;

	try {
		// ʹ�� create_directory ����Ŀ¼
		if (fs::create_directory(dir_path)) {
			std::cout << "Directory created successfully!" << std::endl;
		}
		else {
			std::cout << "Directory already exists or failed to create." << std::endl;
		}
	}
	catch (const fs::filesystem_error& e) {
		std::cout << "Error creating directory: " << e.what() << std::endl;
		return false;
	}

	return true;
}


void writeLog(std::string& log_path, const std::string& message) {

	// ����־�ļ�����׷��ģʽ�򿪣����ļ���������ᴴ����
	std::ofstream logFile(log_path, std::ios::app);

	if (!logFile) {
		std::cerr << "Error opening log file!" << std::endl;
		return;
	}

	// ��ȡ��ǰʱ��
	auto now = std::chrono::system_clock::now();

	// ת��Ϊϵͳʱ�� (time_t)
	std::time_t currentTime = std::chrono::system_clock::to_time_t(now);

	// ʹ�� std::tm �ṹ����������ǰʱ��
	std::tm localTime;
#ifdef _WIN32
	localtime_s(&localTime, &currentTime); // �� Windows ��ʹ�� localtime_s
#else
	localtime_r(&currentTime, &localTime); // �� Unix ϵͳ��ʹ�� localtime_r
#endif

	std::stringstream timestamp;
	timestamp << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S");    //// ��ʽ��ʱ�䣺yyyy-mm-dd hh:mm:ss

	//std::ostringstream timestamp;
	//timestamp << std::put_time(localTime, "%Y-%m-%d %H:%M:%S");  // ��ʽ��ʱ�䣺yyyy-mm-dd hh:mm:ss

	// д����־����
	logFile << "[" << timestamp.str() << "] " << message << std::endl;

	// �ر��ļ�
	logFile.close();
}




///////////////////////////////////////////////////////////
// ��ǩ��ʼ��
///////////////////////////////////////////////////////////
bool readClassNames(std::string labels_txt_file, std::vector<std::string>& classNames)
{
	//std::string labels_txt_file = "E:\\work_program\\lsx_ort_cpp_demo\\zj_onnx_info\\classes.txt";
	//std::string labels_txt_file = "data\\classes.txt";

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		return false;
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return true;
};


//// ������������ std::string ת��Ϊ std::wstring
//std::wstring stringToWString(const std::string& str) {
//	std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
//	return converter.from_bytes(str);
//};



///////////////////////////////////////////////////////////
// API HUB
//1����ʼ��ģ��    init_model_
//2����ȡͼ��ͼ��Ԥ�������    preprocess
//3��ģ������    model_infer
//4���������NMS��    postprocess
//5���ͷ���Դ
///////////////////////////////////////////////////////////
//bool init_model();
bool init_model(std::string& model_path, std::string& cfg_path, std::string& log_path);

bool PLZJDetection(std::string& img_path, std::vector<OutParams>& paramsList_select, int& num);

void FreeOut();




///////////////////////////////////////////////////////////
// ��ʼ��ģ��
// API
///////////////////////////////////////////////////////////
Ort::Session init_model_(const std::string& onnxpath,
	int device);
Ort::Session init_model_(const std::string& onnxpath,
	int device) {

	// ��env���óɾ�̬�ģ��ܽ�� onnxruntime session.Run ���쳣
	static Ort::Env env;
	//Ort::Env env;
	//env = Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "yolov8-rtdetr-onnx");
	env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "yolov8-rtdetr-onnx");
	
	//��־д��
	message = "ONNXRUNTIME Env: yolov8-rtdetr-onnx ORT_LOGGING_LEVEL_VERBOSE";
	writeLog(log_path_all, message);

	// ת��ģ��·��
	//std::wstring onnxpath = stringToWString(onnxpath);
	std::wstring modelPath = std::wstring(onnxpath.begin(), onnxpath.end());

	Ort::SessionOptions sessionOption;
	//sessionOption.SetLogSeverityLevel(0); // ������־����Ϊ��ϸ����
	if (device == 0)
	{
		OrtCUDAProviderOptions cudaOption;
		cudaOption.device_id = device;
		sessionOption.AppendExecutionProvider_CUDA(cudaOption);
	}
	if (device == -1) {
		sessionOption.SetIntraOpNumThreads(1);  // �����߳�����������Դ��ͻ
	}
	//sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	Ort::Session session(env, modelPath.c_str(), sessionOption);

	std::cout << "Model Init Succeed!" << std::endl;
	//��־д��
	message = "Model Init Succeed!";
	writeLog(log_path_all, message);

	return session;

};




//bool init_model(std::string& model_path, std::string& cfg_path, std::string& log_path) {
bool init_model(const char* model_path, const char* cfg_path, const char* log_path) {

	//������־Ŀ¼
	bool create_dir_flag = create_dir(log_path);

	//��ȡ��ǰ�����ڣ� ����log�ļ�·��
	std::string datetime = getCurrentDate();
	std::string log_path_ = log_path;
	log_path_all = log_path_ + "/" + datetime + "_DeepPLLib.txt";

	//��־д��
	message = "create_dir_flag:" + std::to_string(create_dir_flag);
	writeLog(log_path_all, message);


	////��������
	ConfigParser config;
	//std::string config_path = "E:\\work_program\\lsx_ort_cpp_demo\\zj_ort_cpp_demo\\onnx_ort_infer_demo\\x64\\Debug\\data\\deep_zhiju.cfg";
	/*std::string config_path = "data\\deep_zhiju.cfg";*/
	//config.load(config_path);
	config.load(cfg_path);
	using_nms = config.getInt("using_nms");
	conf_threshold = config.getDouble("conf_threshold");
	iou_threshold = config.getDouble("iou_threshold");
	std::string labels_path = config.getString("labels_path");
	//std::string labels_path = "E:\\work_program\\lsx_ort_cpp_demo\\zj_ort_cpp_demo\\onnx_ort_infer_demo\\x64\\Debug\\data\\classes.txt";
	device = config.getInt("device");
	//std::cout << "using_nms: " << using_nms << std::endl;
	//std::cout << "conf_threshold: " << conf_threshold << std::endl;
	//std::cout << "iou_threshold: " << iou_threshold << std::endl;
	//std::cout << "labels_path: " << labels_path << std::endl;
	//std::cout << "device: " << device << std::endl;
	
	//��־д��
	message = "using_nms:" + std::to_string(using_nms);
	writeLog(log_path_all, message);
	message = "conf_threshold:" + std::to_string(conf_threshold);
	writeLog(log_path_all, message);
	message = "iou_threshold:" + std::to_string(iou_threshold);
	writeLog(log_path_all, message);
	message = "labels_path:" + labels_path;
	writeLog(log_path_all, message);
	message = "device:" + device;
	writeLog(log_path_all, message);


	session = init_model_(model_path, device);

	// ���� OpenCV ��־����
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

	return true;
};


///////////////////////////////////////////////////////////
// ͼ��Ԥ����
///////////////////////////////////////////////////////////
bool preprocess(std::string& img_path,
	int input_w,
	int input_h,
	float& x_factor,
	float& y_factor,
	cv::Mat& image_origin,
	cv::Mat& o_image);
bool preprocess(std::string& img_path,
	int input_w,
	int input_h,
	float& x_factor,
	float& y_factor,
	cv::Mat& image_origin,
	cv::Mat& o_image) {
	// ��ȡͼ��
	cv::Mat image = cv::imread(img_path);
	image_origin = image.clone();
	if (image.empty()) {
		std::cerr << "Error loading image!" << std::endl;
		return false;  // ͼ�����ʧ�ܣ�����false
	}

	// ����ͼ����������
	x_factor = image.cols / static_cast<float>(input_w);
	y_factor = image.rows / static_cast<float>(input_h);

	// BGR ת RGB
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// resize ͼ��
	cv::resize(image, image, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

	// ��һ����1/255�� ͼ��ֵ��Χ 0-1
	image.convertTo(o_image, CV_32F, 1.0/255.0);

	//// ������������
	//size_t tpixels = input_h * input_w * 3;
	//std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
	//auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	//// ֱ��ʹ��ͼ�����ݴ�����������
	//input_tensor = Ort::Value::CreateTensor<float>(allocator_info, image.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());

	return true;
};



///////////////////////////////////////////////////////////
// ģ������
///////////////////////////////////////////////////////////
//bool model_infer(Ort::Session& session,
//	const std::array<const char*, 1>& inputNames,
//	const std::array<const char*, 1>& outNames,
//	const Ort::Value& input_tensor,
//	std::vector<Ort::Value>& ort_outputs);
bool model_infer(Ort::Session& session,
	const std::array<const char*, 1>& inputNames,
	const std::array<const char*, 1>& outNames,
	const Ort::Value& input_tensor,
	std::vector<Ort::Value>& ort_outputs) {

	//std::cout << "Inference -->  " << std::endl;
	try {
		// ִ������
		ort_outputs = session.Run(Ort::RunOptions{ nullptr },
			inputNames.data(),
			&input_tensor,
			1,
			outNames.data(),
			outNames.size());
	}
	catch (const std::exception& e) {
		std::cout << "Inference failed: " << e.what() << std::endl;
		return false;
	}

	return true;

};



///////////////////////////////////////////////////////////
// ���ݺ���
///////////////////////////////////////////////////////////
//bool postprocess(std::vector<Ort::Value>& ort_outputs,
//	int input_h,
//	int input_w,
//	int output_h,
//	int output_w,
//	double x_factor,
//	double y_factor,
//	double conf_threshold,
//	double iou_threshold,
//	int using_nms,
//	std::vector<OutParams>& paramsList_select,
//	int& num);
bool postprocess(std::vector<Ort::Value>& ort_outputs,
	int input_h,
	int input_w,
	int output_h,
	int output_w,
	double x_factor,
	double y_factor,
	double conf_threshold,
	double iou_threshold,
	int using_nms,
	std::vector<OutParams>& paramsList_select,
	int& num) {

	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(output_h, output_w, CV_32F, (float*)pdata);
	cv::Mat det_output = dout.t(); // 300x15 (11+4)

	// post-process
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<OutParams> paramsList;

	//��־д��
	message = "num_origin:" + std::to_string(det_output.cols);
	writeLog(log_path_all, message);

	for (int i = 0; i < det_output.cols; i++) {
		cv::Mat classes_scores = det_output.col(i).rowRange(4, 15);
		classes_scores.convertTo(classes_scores, CV_32F);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// ���Ŷ�
		if (score > conf_threshold)
		{
			//cx, cy, ow, oh
			float cx = det_output.at<float>(0, i) * input_w;
			float cy = det_output.at<float>(1, i) * input_h;
			float ow = det_output.at<float>(2, i) * input_w;
			float oh = det_output.at<float>(3, i) * input_h;
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.y);
			confidences.push_back(score);

			//std::cout << "box: " << box << std::endl;
			//std::cout << "score: " << score << std::endl;
			//std::cout << "classIdPoint.y: " << classIdPoint.y << std::endl;

			OutParams tmp_param;
			tmp_param._bbX1 = box.x;
			tmp_param._bbY1 = box.y;
			tmp_param._bbX2 = box.x + box.width;
			tmp_param._bbY2 = box.y + box.height;
			tmp_param._confidence = score;
			tmp_param._id = classIdPoint.y;
			paramsList.push_back(tmp_param);
		}
	}

	//��־д��
	message = "num_no_nms:" + std::to_string(paramsList.size());
	writeLog(log_path_all, message);


	// NMS
	std::vector<int> indexes;
	//std::vector<OutParams> paramsList_select;
	if (using_nms) {
		cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, indexes);

		for (size_t i = 0; i < indexes.size(); i++) {
			int index = indexes[i];
			int idx = classIds[index];
			double score_idx = confidences[index];
			cv::Rect box = boxes[index];

			OutParams tmp_param;
			tmp_param._bbX1 = box.x;
			tmp_param._bbY1 = box.y;
			tmp_param._bbX2 = box.x + box.width;
			tmp_param._bbY2 = box.y + box.height;
			tmp_param._confidence = score_idx;
			tmp_param._id = idx;
			paramsList_select.push_back(tmp_param);
		}
	}
	else {
		paramsList_select = paramsList;
	}

	num = paramsList_select.size();

	//��־д��
	message = "using_nms:" + std::to_string(using_nms);
	writeLog(log_path_all, message);

	return true;

};


//std::vector<OutParams> ת OutParams**
bool vectorToOutParamsPointer(std::vector<OutParams>& vec, OutParams**& pOutims) {
	// ����ָ�� OutParams* ��ָ������
	int size = vec.size();
	pOutims = new OutParams * [size];
	//*pOutims = new OutParams[size];

	// Ϊÿ�� OutParams Ԫ�ط����ڴ棬�������ݸ���
	for (int i = 0; i < size; ++i) {
		pOutims[i] = new OutParams(vec[i]);  // ����Ԫ�ص��·�����ڴ�
	}

	return true;
}


OutParams* vectorToArray(const std::vector<OutParams>& vec) {
	// ��̬���� OutParams ����
	OutParams* array = new OutParams[vec.size()];
	// ��������
	for (size_t i = 0; i < vec.size(); ++i) {
		array[i] = vec[i];
	}
	return array;
}


///////////////////////////////////////////////////////////
// ��������ͼƬ
// API
///////////////////////////////////////////////////////////
bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num) {

	std::string img_path2(img_path);    // ʹ�� std::string �Ĺ��캯��
	std::cout << "PLZJDetection img_path:" << img_path << std::endl;
	//��־д��
	message = "img_path:" + img_path2;
	writeLog(log_path_all, message);

	// ͼ��Ԥ���� ����
	cv::Mat input_image;    // ������ͼ��
	cv::Mat frame;    // ԭʼ��ͼ��
	bool preprocess_flag = preprocess(img_path2,
		input_w,
		input_h,
		x_factor,
		y_factor,
		frame,
		input_image);
	//std::cout << "input_image cols:" << input_image.cols << std::endl;
	//��־д��
	message = "preprocess_flag:" + std::to_string(preprocess_flag);
	writeLog(log_path_all, message);



	//ͼ��ת��tensor
	cv::Mat blob = cv::dnn::blobFromImage(input_image);

	// ������������ shape
	size_t tpixels = input_h * input_w * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };
	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	// ֱ��ʹ��ͼ�����ݴ�����������
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	//std::unique_ptr<Ort::Value> input_tensor = std::make_unique<Ort::Value>(Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size()));
	//��־д��
	message = "create input_tensor flag succeed!";
	writeLog(log_path_all, message);

	//��ȡģ�͵���������Ȳ���
	std::vector<std::string> input_node_names;
	std::vector<std::string> output_node_names;
	size_t numInputNodes = session.GetInputCount();
	size_t numOutputNodes = session.GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	input_node_names.reserve(numInputNodes);

	// ��ȡ������Ϣ
	for (int i = 0; i < numInputNodes; i++) {
		auto input_name = session.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		Ort::TypeInfo input_type_info = session.GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		//std::cout << "input format: NxCxHxW = " << input_dims[0] << "x" << input_dims[1] << "x" << input_dims[2] << "x" << input_dims[3] << std::endl;
	}

	// ��ȡ�����Ϣ
	int output_h = 0;
	int output_w = 0;
	Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
	auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	auto output_dims = output_tensor_info.GetShape();
	output_h = output_dims[1]; // 300
	output_w = output_dims[2]; // 15
	//std::cout << "output format : HxW = " << output_dims[1] << "x" << output_dims[2] << std::endl;
	for (int i = 0; i < numOutputNodes; i++) {
		auto out_name = session.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(out_name.get());
	}
	//std::cout << "input name: " << input_node_names[0] << " output name: " << output_node_names[0] << std::endl;

	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	const std::array<const char*, 1> outNames = { output_node_names[0].c_str() };


	//ģ������
	//std::cout << "model_infer strat ... " << std::endl;
	//��־д��
	message = "model_infer strat ...";
	writeLog(log_path_all, message);
	std::vector<Ort::Value> ort_outputs;
	ort_outputs = session.Run(Ort::RunOptions{ nullptr },
		inputNames.data(),
		&input_tensor,
		inputNames.size(),
		outNames.data(),
		outNames.size());

	//��־д��
	message = "onnx model_infer end!";
	writeLog(log_path_all, message);

	//�������NMS��
	std::vector<OutParams> paramsList_select;
	auto postprocess_flag = postprocess(ort_outputs,
		input_h,
		input_w,
		output_h,
		output_w,
		x_factor,
		y_factor,
		conf_threshold,
		iou_threshold,
		using_nms,
		paramsList_select,
		num);
	//std::cout << "results: " << results << std::endl;
	//��־д��
	message = "postprocess_flag:" + std::to_string(postprocess_flag);
	writeLog(log_path_all, message);

	//��־д��
	message = "num_nms:" + std::to_string(num);
	writeLog(log_path_all, message);

	//// �� std::vector<OutParams> ת��Ϊ OutParams** ���͵�ָ��
	//bool flag = vectorToOutParamsPointer(paramsList_select, pOutims);

	// ת��Ϊ OutParams*
	pOutims = vectorToArray(paramsList_select);

	return true;
};



///////////////////////////////////////////////////////////
// �ͷ���Դ
///////////////////////////////////////////////////////////
void FreeOut() {

	session.release();
	//delete onnxpath;
	//delete config_path;
	//delete input_tensor;
	//delete x_factor;
	//delete y_factor;
	//delete input_w;
	//delete input_h;
	//delete using_nms;
	//delete conf_threshold;
	//delete iou_threshold;
	//delete device;

	//��־д��
	message = "FreeOut succeed!";
	writeLog(log_path_all, message);

};


///////////////////////////////////////////////////////////
// �ͷ�
///////////////////////////////////////////////////////////
//void FreeOutParams(OutParams** pOutims, int num) {
//
//	// �ͷ�ÿ��������ڴ�
//	for (int i = 0; i < num; ++i) {
//		delete pOutims[i];  // �ͷ�ÿ�� OutParams ����
//	}
//
//	// �ͷ�ָ��ָ��������ڴ�
//	delete[] pOutims;  // �ͷ� OutParams* ����
//}


void FreeOutParams(OutParams*& pOutims, int num) {

	////// �ͷ�ÿ��������ڴ�
	//for (int i = 0; i < num; ++i) {
	//	delete pOutims[i];  // �ͷ�ÿ�� OutParams ����
	//}

	// �ͷ�ָ��ָ��������ڴ�
	delete[] pOutims;  // �ͷ� OutParams* ����
	pOutims = nullptr;

}





