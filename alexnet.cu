#include <cudnn.h>
#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cmath> 
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include "error_util.h"

#define IMAGE_H 227
#define IMAGE_W 227

// weights file
const char *conv1_b_bin = "model_weights/conv1_b_0_weights.bin";
const char* conv1_w_bin = "model_weights/conv1_w_0_weights.bin";
const char* conv2_b_bin = "model_weights/conv2_b_0_weights.bin";
const char* conv2_w_bin = "model_weights/conv2_w_0_weights.bin";
const char* conv3_b_bin = "model_weights/conv3_b_0_weights.bin";
const char* conv3_w_bin = "model_weights/conv3_w_0_weights.bin";
const char* conv4_b_bin = "model_weights/conv4_b_0_weights.bin";
const char* conv4_w_bin = "model_weights/conv4_w_0_weights.bin";
const char* conv5_b_bin = "model_weights/conv5_b_0_weights.bin";
const char* conv5_w_bin = "model_weights/conv5_w_0_weights.bin";
const char* fc1_b_bin = "model_weights/fc6_b_0_fc.bin";
const char* fc1_w_bin = "model_weights/fc6_w_0_fc.bin";
const char* fc2_b_bin = "model_weights/fc7_b_0_fc.bin";
const char* fc2_w_bin = "model_weights/fc7_w_0_fc.bin";
const char* fc3_b_bin = "model_weights/fc8_b_0_fc.bin";
const char* fc3_w_bin = "model_weights/fc8_w_0_fc.bin";

// image files
const char* image_goldfish = "images/n01443537_goldfish.JPEG";
const char* image_hen = "images/n01514859_hen.JPEG";
const char* image_scorpion = "images/n01770393_scorpion.JPEG";


// load image and resize image function
cv::Mat load_image(const char* image_path){
    cv:: Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if(image.empty()){
        std::cerr << "Error loading image!" << std::endl;
        return image;
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(227,227));
    resized_image.convertTo(resized_image, CV_32FC3, 1.0/255.0);

    return resized_image;
}

// read binary weights file
template<class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h){
    // create an ifstream object named file and open the file specified by fname in binary mode
    std::ifstream file(fname, std::ios::binary);
    std::stringstream error_s;
    if(!file){

        std::cerr << "Error opening file: " << fname << std::endl;
        FatalError(error_s.str());
    }

    // moves the 'get' pointer (the position where the next read will occur) to end of file
    file.seekg(0, std::ios::end);
    // tellg() returns current position of 'get' pointer in input stream. It tells you how many bytes into file you currently are
    std::streamsize file_size = file.tellg();
    // resets the 'get' pointer back to beginning of file
    file.seekg(0, std::ios::beg);
    std::cout << "Loading binary file " << fname << std::endl;
    // check if file is FP16
    bool is_fp16 = (file_size == size*sizeof(__half));

    // if constexpr(std:: is_same<value_type, __half>::value){
    if(is_fp16){
        std::cout << "Detected FP16 format for binary file, converting to FP32 format..." << std::endl;
        // temporary buffer for FP16 data
        __half* tmp_fp16 = new __half[size];
        int size_b = size*sizeof(__half);

        if(!file.read((char*) tmp_fp16, size_b)){
            error_s <<"Error reading file " << fname;
            FatalError(error_s.str());
        }

        for(int i = 0; i < size; i++){
            data_h[i] = __half2float(tmp_fp16[i]);
        }

        delete [] tmp_fp16;

    } else {
        // if data is stored in fp32 and data_h is fp32
        int size_b = size*sizeof(float);
        // read function takes a char* so data_tmp is cast to char
        std::cout << "Attempting to read " << fname << " with size: " << size << " floats (" << size_b << " bytes)" << std::endl;

        if(!file.read((char*) data_h, size_b)){
            error_s << "Error reading file " << fname;
            FatalError(error_s.str());
        }
    }

}

// read and allocate memory for layer weights
template<class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d){

    // allocate host memory
    *data_h = new value_type[size];
    // read binary data from file to host memory
    readBinaryFile<value_type>(fname, size, *data_h);
    checkCUDA(cudaMalloc(data_d, size*sizeof(value_type)));
    checkCUDA(cudaMemcpy(*data_d, *data_h, size*sizeof(value_type), cudaMemcpyHostToDevice));
}

typedef enum {
    FP16_HOST = 0,
    FP16_CUDA = 1,
    FP16_CUDNN = 2
} fp16Import;


template <class value_type>
struct Layer{

    // Data members
    fp16Import fp16_mode; // declares a variable fp16Import of type fp16Import
    int inputs, outputs, kernel_dim;
    value_type *data_h, *data_d, *bias_h, *bias_d;

    
    // default constructor, initializes instance of layer with default values
    Layer() : data_h(NULL), data_d(NULL), bias_h(NULL), bias_d(NULL),
              inputs(0), outputs(0), kernel_dim(0), fp16_mode(FP16_HOST) {}

    //  parameterized constructor
    Layer::Layer(int _inputs, int _outputs, int _kernel_dim, const char* weights_file, const char* bias_file, fp16Import _fp16Import = FP16_HOST)
        :inputs(_inputs), outputs(_outputs), kernel_dim(_kernel_dim), fp16_mode(_fp16Import){
        
        std::string weights_path, bias_path;
        weights_path = weights_file; bias_path = bias_file;
        
        // check if the weights belong to fully connected layer
        int weight_size = (_kernel_dim > 0) ? inputs*outputs*kernel_dim*kernel_dim : inputs*outputs;
        readAllocInit(weights_path.c_str(), weight_size, &data_h, &data_d);
        readAllocInit(bias_path.c_str(), outputs, &bias_h, &bias_d);
    }

    // destructor
    ~Layer(){
        if(data_h != NULL) delete[] data_h;
        if(bias_h != NULL) delete[] bias_h;
        if(data_d != NULL) checkCUDA(cudaFree(data_d));
        if(bias_d != NULL) checkCUDA(cudaFree(bias_d));
    }

    // private methods
    private:

        void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d){
            readAllocMemcpy<value_type>(fname, size, data_h, data_d);
        }
};

void setTensorDesc(cudnnTensorDescriptor_t &tensorDesc, cudnnTensorFormat_t &format, cudnnDataType_t &dataType, int n, int c, int h, int w){

    checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc, format, dataType, n, c, h, w));
}


template<class value_type>
class network{

    cudnnHandle_t cudnn;
    cublasHandle_t cublas;
    int convAlgorithm;
    int returnedAlgoCount;
    const float scale_alpha =1, scale_beta = 0;
    cudnnDataType_t dataType;
    cudnnTensorFormat_t format;
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc, biasTensorDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnPoolingDescriptor_t poolDesc;
    cudnnActivationDescriptor_t activDesc;
    cudnnLRNDescriptor_t lrnDesc;

    void createHandles(){

        // create handles
        checkCUDNN(cudnnCreate(&cudnn));
        checkCUBLAS(cublasCreate(&cublas));
        checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
        checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
        checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
        checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
        checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
        checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
        checkCUDNN(cudnnCreateLRNDescriptor(&lrnDesc));
    }

    void destroyHandles(){
        
        // destroy handles
        checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
        checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
        checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
        checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
        checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
        checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
        checkCUDNN(cudnnDestroyLRNDescriptor(lrnDesc));
        checkCUDNN(cudnnDestroy(cudnn));
        checkCUBLAS(cublasDestroy(cublas));
    }
    
    public :

    network(){

        convAlgorithm = -1;
        format = CUDNN_TENSOR_NHWC;
        dataType = CUDNN_DATA_FLOAT;
        createHandles();
    };

    ~network(){
        destroyHandles();
    }

    void resize(int size, value_type **data){
        if(*data != NULL){
            checkCUDA(cudaFree(*data));
        }
        checkCUDA(cudaMalloc(data, size*sizeof(value_type)));
    }

    void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t &algo){
        convAlgorithm = (int) algo;
    }

    void addBias(const cudnnTensorDescriptor_t dstTensorDesc, const Layer<value_type> &layer, int c, value_type *data){
        
        setTensorDesc(biasTensorDesc, format, dataType, 1, c, 1, 1);
        checkCUDNN(cudnnAddTensor(cudnn, &scale_alpha, biasTensorDesc, layer.bias_d, &scale_alpha, dstTensorDesc, data));
    }

    void LRNForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData) {

        resize(n*c*h*w, dstData);
        setTensorDesc(srcTensorDesc, format, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, format, dataType, n, c, h, w);

        checkCUDNN(cudnnSetLRNDescriptor(lrnDesc, 
                                        5,          // local size
                                        0.0001f,    // alpha
                                        0.75f,      // beta
                                        2.0f));     // k

        checkCUDNN(cudnnLRNCrossChannelForward(cudnn, lrnDesc, CUDNN_LRN_CROSS_CHANNEL_DIM1, &scale_alpha, srcTensorDesc, srcData, &scale_beta, dstTensorDesc, *dstData));
    }

    void convForward(const Layer<value_type> &conv, int &n, int &c, int &h, int &w, int pad, int stride, value_type* srcData, value_type** dstData){

        cudnnConvolutionFwdAlgo_t algo;

        setTensorDesc(srcTensorDesc, format, dataType, n, c, h, w);
        checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc, dataType, format, conv.outputs, conv.inputs, conv.kernel_dim, conv.kernel_dim)); 
        checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT)); 
        checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc, srcTensorDesc, filterDesc, &n, &c, &h, &w));
    
        setTensorDesc(dstTensorDesc, format, dataType, n, c, h, w);
        cudnnConvolutionFwdAlgoPerf_t perfResults[1];
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, 1, &returnedAlgoCount, perfResults));

        resize(n*c*h*w, dstData);
        size_t workspaceSizeBytes;
        void* workspace_d = nullptr;
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, srcTensorDesc, filterDesc, convDesc, dstTensorDesc, perfResults[0].algo, &workspaceSizeBytes));

        if (workspaceSizeBytes != 0){
            checkCUDA(cudaMalloc(&workspace_d, workspaceSizeBytes));
        }

        checkCUDNN(cudnnConvolutionForward(cudnn, &scale_alpha, srcTensorDesc, srcData, filterDesc, conv.data_d, convDesc, perfResults[0].algo, workspace_d, workspaceSizeBytes, &scale_beta, dstTensorDesc, *dstData));
        addBias(dstTensorDesc, conv, c, *dstData);

        if(workspaceSizeBytes != 0){
            checkCUDA(cudaFree(workspace_d));
        }
    }

    void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData){
        
        resize(n*c*h*w, dstData);
        setTensorDesc(srcTensorDesc, format, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, format, dataType, n, c, h, w);
        checkCUDNN(cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
        checkCUDNN(cudnnActivationForward(cudnn, activDesc, &scale_alpha, srcTensorDesc, srcData, &scale_beta, dstTensorDesc, *dstData));
    }

    void poolForward(int &n, int &c, int &h, int &w, value_type *srcData, value_type **dstData){

        checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 3, 3, 0, 0, 2, 2));
        setTensorDesc(srcTensorDesc, format, dataType, n, c, h, w);
        checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc, srcTensorDesc, &n, &c, &h, &w));
        setTensorDesc(dstTensorDesc, format, dataType, n, c, h, w);
        resize(n*c*h*w, dstData);
        checkCUDNN(cudnnPoolingForward(cudnn, poolDesc, &scale_alpha, srcTensorDesc, srcData, &scale_beta, dstTensorDesc, *dstData));
    }

    void fullyConnectedForward(const Layer<value_type> &fc, int &n, int &c, int &h, int &w, value_type* srcData, value_type** dstData){

        int input_dim = c*h*w;
        int output_dim = fc.outputs;
        resize(output_dim, dstData);


        checkCUDA(cudaMemcpy(*dstData, fc.bias_d, output_dim*sizeof(value_type), cudaMemcpyDeviceToDevice));
        checkCUBLAS(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, 1, input_dim, &scale_alpha, fc.data_d, input_dim, srcData, input_dim, &scale_alpha, *dstData, output_dim));
        h =1; w=1; c= output_dim;
    }

    void softmaxForward(int n, int c, int h, int w, value_type *srcData, value_type** dstData ){

        resize(n*c*h*w, dstData);
        setTensorDesc(srcTensorDesc, format, dataType, n, c, h, w);
        setTensorDesc(dstTensorDesc, format, dataType, n, c, h, w);
        checkCUDNN(cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &scale_alpha, srcTensorDesc, srcData, &scale_beta, dstTensorDesc, *dstData));
    }
    
    void classify_example(const char* fname, const Layer<value_type> &conv1, 
                         const Layer<value_type> &conv2, const Layer<value_type> &conv3,
                         const Layer<value_type> &conv4, const Layer<value_type> &conv5,
                         const Layer<value_type> &fc6, const Layer<value_type> &fc7, 
                         const Layer<value_type> &fc8)
    {
        int n, c, h, w;
        value_type *srcData = NULL, *dstData = NULL;
        value_type imgData_h[IMAGE_H*IMAGE_W*3];

        cv::Mat resized_image = load_image(fname);

        if(resized_image.empty()) {
            std::cerr << "Error: Could not load the image" << std::endl;
            return;
        }

        // .ptr<type>() returns a pointer to first element in cv::Mat, covering entire flattened array of pixel data for all rows
        std::memcpy(imgData_h, resized_image.ptr<value_type>(), IMAGE_H*IMAGE_W*3*sizeof(value_type));

        std::cout <<"Performing forward propagation ...\n";
        checkCUDA(cudaMalloc(&srcData, IMAGE_H*IMAGE_W*3*sizeof(value_type)));
        checkCUDA(cudaMemcpy(srcData, imgData_h, IMAGE_H*IMAGE_W*3*sizeof(value_type), cudaMemcpyHostToDevice));
        n = 1; c = 3; h = IMAGE_H; w= IMAGE_W;
        convForward(conv1, n, c, h, w, 2, 4, srcData, &dstData);
        LRNForward(n, c, h, w, dstData, &srcData);
        activationForward(n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);
        std::cout << "First Layer Completed!" << std::endl;

        convForward(conv2, n, c, h, w, 2, 1, srcData, &dstData);
        LRNForward(n, c, h, w, dstData, &srcData);
        activationForward(n, c, h, w, srcData, &dstData);
        poolForward(n, c, h, w, dstData, &srcData);
        std::cout << "Second Layer Completed!" << std::endl;

        convForward(conv3, n, c, h, w, 1, 1, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
        std::cout << "Third Layer Completed!" << std::endl; 

        convForward(conv4, n, c, h, w, 1, 1, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
        std::cout << "Fourth Layer Completed!" << std::endl;

        convForward(conv5, n, c, h, w, 1, 1, srcData, &dstData);
        activationForward(n, c, h, w, dstData, &srcData);
        poolForward(n, c, h, w, srcData, &dstData);
        std::cout << "Fifth Layer Completed!" << std::endl;

        fullyConnectedForward(fc6, n, c, h, w, dstData, &srcData);
        activationForward(n, c, h, w, srcData, &dstData);

        fullyConnectedForward(fc7, n, c, h, w, dstData, &srcData);
        activationForward(n, c, h, w, srcData, &dstData);

        fullyConnectedForward(fc8, n, c, h, w, dstData, &srcData);
        softmaxForward(n, c, h, w, srcData, &dstData);

        checkCUDA(cudaDeviceSynchronize());

        const int max_digits = 1000;
        value_type result[max_digits];
        checkCUDA(cudaMemcpy(result, dstData, max_digits*sizeof(value_type), cudaMemcpyDeviceToHost));
        int id = 0;
        for(int i = 1; i < max_digits; i++){
            if(result[id] < result[i]){
                id = i;
            }
        }

        std::cout << "Index of maximum weight: " << id << std::endl;
    } 

};


int main(){

    network<float> alexnet;

    // all data structures, data_h, data_d, bias_h, bias_d are allocated and handled in fp32. Fp16_mode is ignored
    Layer<float> conv1(3, 96, 11, conv1_w_bin, conv1_b_bin, FP16_HOST);
    Layer<float> conv2(96, 256, 5, conv2_w_bin, conv2_b_bin, FP16_HOST);
    Layer<float> conv3(256, 384, 3, conv3_w_bin, conv3_b_bin, FP16_HOST);
    Layer<float> conv4(384, 384, 3, conv4_w_bin, conv4_b_bin, FP16_HOST);
    Layer<float> conv5(384, 256, 3, conv5_w_bin, conv5_b_bin, FP16_HOST);
    Layer<float>  fc6(9216, 4096, 0,  fc1_w_bin, fc1_b_bin, FP16_HOST);
    Layer<float> fc7(4096, 4096, 0, fc2_w_bin, fc2_b_bin, FP16_HOST);
    Layer<float> fc8(4096, 1000, 0, fc3_w_bin, fc3_b_bin, FP16_HOST);
    alexnet.classify_example(image_scorpion, conv1, conv2, conv3, conv4, conv5, fc6, fc7, fc8);

    return 0;
}

