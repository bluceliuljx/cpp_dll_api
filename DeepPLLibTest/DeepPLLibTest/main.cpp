#include <iostream>
#include <Windows.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;


typedef struct OutParams {
    int _bbX1;
    int _bbY1;
    int _bbX2;
    int _bbY2;
    float _confidence;
    int _id;
    int _img_idx;
};


// 配置解析类
class ConfigParser {
public:
    // 加载配置文件
    bool load(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Unable to open config file: " << filepath << std::endl;
            return false;
        }

        std::string line;
        while (std::getline(file, line)) {
            line = trim(line);

            // 忽略注释行和空行
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // 查找 '=' 符号分隔键和值
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

    // 获取字符串值
    std::string getString(const std::string& key, const std::string& default_value = "") const {
        auto it = config_data.find(key);
        if (it != config_data.end()) {
            return it->second;
        }
        return default_value;
    }

    // 获取整数值
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

    // 获取浮点值
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

    // 获取布尔值
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

    // 去除字符串前后空格
    static std::string trim(const std::string& str) {
        size_t start = str.find_first_not_of(" \t");
        size_t end = str.find_last_not_of(" \t");
        return (start == std::string::npos) ? "" : str.substr(start, end - start + 1);
    }
};

///////////////////////////////////////////////////////////
// 标签初始化
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


int main() {

    std::cout << "test api start!" << std::endl;

    // 加载 DLL
    HMODULE hDll = LoadLibrary(L"DeepPLLib.dll");
    if (hDll == NULL) {
        std::cerr << "Failed to load DLL!" << std::endl;
        return -1;
    }

    // TestAPI
    typedef void (*TestAPI_func)();
    TestAPI_func TestAPI = (TestAPI_func)GetProcAddress(hDll, "TestAPI");

    //初始化模型
    //DeepPL_DLL_API bool init_model();
    //init_model(std::string & model_path, std::string & cfg_path, std::string & log_path);
    //typedef bool(*init_model)(std::string& model_path, std::string& cfg_path, std::string& log_path);
    typedef bool(*init_model)(const char* model_path, const char* cfg_path, const char* log_path);
    init_model init_model_func = (init_model)GetProcAddress(hDll, "init_model");

     //分析函数
    //DeepPL_DLL_API bool PLZJDetection(std::string & img_path, std::vector<OutParams>&pOutims, int& num);
    //typedef bool(*PLZJDetection)(const char* img_path, std::vector<OutParams>& pOutims, int& num);
    //DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams*& pOutims, int& num);
    typedef bool(*PLZJDetection)(const char* img_path, OutParams*& pOutims, int& num);
    PLZJDetection PLZJDetection_func = (PLZJDetection)GetProcAddress(hDll, "PLZJDetection");


    //图像预处理
    //bool preprocess(std::string& img_path,
    //    int input_w,
    //    int input_h,
    //    float& x_factor,
    //    float& y_factor,
    //    cv::Mat & image_origin,
    //    cv::Mat & o_image);
    typedef bool(*preprocess)(std::string& img_path,
            int input_w,
            int input_h,
            float& x_factor,
            float& y_factor,
            cv::Mat & image_origin,
            cv::Mat & o_image);
    preprocess preprocess_func = (preprocess)GetProcAddress(hDll, "preprocess");


    //DeepPL_DLL_API void FreeOutParams(OutParams** pOutims, int num);
    typedef void (*FreeOutParams)(OutParams*& pOutims, int num);
    FreeOutParams FreeOutParams_func = (FreeOutParams)GetProcAddress(hDll, "FreeOutParams");


    //DeepPL_DLL_API void FreeOut();
    typedef void (*FreeOut)();
    FreeOut FreeOut_func = (FreeOut)GetProcAddress(hDll, "FreeOut");



    if (TestAPI == NULL) {
        std::cerr << "Failed to get function addresses!" << std::endl;
        FreeLibrary(hDll);
        return -1;
    }

    // 调用 DLL 函数
    TestAPI();


    //模型初始化
    std::cout << "init_model_func!" << std::endl;
    //std::string dir_f = "E:/work_program/lsx_ort_cpp_demo/zj_ort_cpp_dll/deepPL/DeepPLLibTest/x64/Release";
    std::string dir_f = "./";
    std::string model_path = dir_f + "/" + "data/zhiju.onnx";
    std::string cfg_path = dir_f + "/" + "data/deep_zhiju.cfg";
    std::string log_path = dir_f + "/" + "LOG";
    const char* model_path2 = model_path.c_str();
    const char* cfg_path2 = cfg_path.c_str();
    const char* log_path2 = log_path.c_str();
    bool init_model_func_statue = init_model_func(model_path2, cfg_path2, log_path2);
    std::cout << "init_model_func_statue:" << init_model_func_statue << std::endl;

    ////解析配置
    ConfigParser config;
    //std::string config_path = "E:/work_program/lsx_ort_cpp_demo/zj_ort_cpp_dll/deepPL/DeepPLLibTest/x64/Release/DeepPLLibTest.cfg";
    std::string config_path = "DeepPLLibTest.cfg";
    config.load(config_path);

    int infer_n = 1;    // 测试推理次数
    for (int iii = 0; iii < infer_n;iii++) {
        std::cout << "infer_n:" << iii << std::endl;

        std::string img_path = config.getString("img_path");
        std::cout << "img_path:" << img_path << std::endl;
        const char* img_path2 = img_path.c_str();
        std::cout << "img_path2:" << img_path2 << std::endl;

        //图像推理
        int64 start = cv::getTickCount();
        //std::string img_path = "E:\\work_program\\lsx_ort_cpp_demo\\zj_ort_cpp_dll\\deepPL\\DeepPLLibTest\\x64\\Debug\\images\\20250103_140804_777.jpg";
        int input_w = 1280;
        int input_h = 1280;
        float x_factor;
        float y_factor;
        cv::Mat frame;
        cv::Mat o_image;
        bool preprocess_func_statue = preprocess_func(img_path,
            input_w,
            input_h,
            x_factor,
            y_factor,
            frame,
            o_image);
        std::cout << "preprocess_func_statue:" << preprocess_func_statue << std::endl;

        std::cout << "o_image rows:" << o_image.rows << std::endl;

        OutParams* pOutims = nullptr;
        int num = 0;
        bool PLZJDetection_func_statue = PLZJDetection_func(img_path2, pOutims, num);
        std::cout << "PLZJDetection_func_statue:" << PLZJDetection_func_statue << std::endl;


        //可视化
        //std::string labels_txt_file = "E:/work_program/lsx_ort_cpp_demo/zj_ort_cpp_dll/deepPL/DeepPLLibTest/x64/Release/data/classes.txt";
        std::string labels_txt_file = "data/classes.txt";
        std::vector<std::string> classNames;
        readClassNames(labels_txt_file, classNames);
        std::cout << "classNames 0:" << classNames[0] << std::endl;

        std::cout << "num:" << num << std::endl;
        std::cout << "pOutims sizeof:" << sizeof(pOutims) << std::endl;
        std::cout << "pOutims[0] sizeof:" << sizeof(pOutims[0]) << std::endl;
        std::cout << "*pOutims sizeof:" << sizeof(*pOutims) << std::endl;
        std::cout << "pOutims:" << pOutims << std::endl;

        // 创建一个窗口，指定窗口自动调整大小
        cv::namedWindow("YOLO RTDETR ONNXRUNTIME DEMO", cv::WINDOW_NORMAL);

        std::cout << "num_id" << "\t" << "bbX1" << "\t" << "bbY1" << "\t" << "bbX2" << "\t" << "bbY2" << "\t" << "label" << "\t" << "score" << std::endl;
        for (size_t i = 0; i < num; i++) {
            cv::Rect box;
            //box.x = pOutims[i]->_bbX1;
            //box.y = pOutims[i]->_bbY1;
            //box.width = pOutims[i]->_bbX2 - pOutims[i]->_bbX1;
            //box.height = pOutims[i]->_bbY2 - pOutims[i]->_bbY1;
            //int label = pOutims[i]->_id;
            //double score = pOutims[i]->_confidence;

            //std::cout << i << "\t"
            //    << pOutims[i]->_bbX1 << "\t"
            //    << pOutims[i]->_bbY1 << "\t"
            //    << pOutims[i]->_bbX2 << "\t"
            //    << pOutims[i]->_bbY2 << "\t"
            //    << pOutims[i]->_id << "\t"
            //    << pOutims[i]->_confidence << std::endl;

            box.x = pOutims[i]._bbX1;
            box.y = pOutims[i]._bbY1;
            box.width = pOutims[i]._bbX2 - pOutims[i]._bbX1;
            box.height = pOutims[i]._bbY2 - pOutims[i]._bbY1;
            int label = pOutims[i]._id;
            double score = pOutims[i]._confidence;

            std::cout << i << "\t"
                << pOutims[i]._bbX1 << "\t"
                << pOutims[i]._bbY1 << "\t"
                << pOutims[i]._bbX2 << "\t"
                << pOutims[i]._bbY2 << "\t"
                << pOutims[i]._id << "\t"
                << pOutims[i]._confidence << std::endl;

            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 1.3, 8);
            cv::rectangle(frame, cv::Point(box.tl().x, box.tl().y - 20),
                cv::Point(box.br().x, box.tl().y), cv::Scalar(0, 255, 255), -1);
            putText(frame, classNames[label] + " " + std::to_string(score).substr(0, 4), cv::Point(box.tl().x, box.tl().y), cv::FONT_HERSHEY_PLAIN, 1.3, cv::Scalar(255, 0, 0), 1.5, 8);
            cv::imshow("YOLO RTDETR ONNXRUNTIME DEMO", frame);
        }

        // 计算FPS render it
        float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("YOLO RTDETR ONNXRUNTIME DEMO", frame);
        cv::waitKey(0);


        //删除pOutims 资源
            FreeOutParams_func(pOutims, num);
    }
    


    //释放资源
    FreeOut_func();


    // 卸载 DLL
    FreeLibrary(hDll);

    std::cout << "TEST API END!" << std::endl;
}





///////////////////////////////////////////////////////////
//流水线治具调用步骤
///////////////////////////////////////////////////////////
/*
1、初始化
    DeepPL_DLL_API bool init_model(const char* model_path, const char* cfg_path, const char* log_path);
2、分析
    2.1、分析单张图像
        DeepPL_DLL_API bool PLZJDetection(const char* img_path, OutParams * *&pOutims, int& num);
    2.2、每次分析完成单纯单张图像后，要释放pOutims
        DeepPL_DLL_API void FreeOutParams(OutParams * pOutim);
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

