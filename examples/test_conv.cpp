/*
  Description : copy from test_dw_conv_general.cpp
  Author : lq
  Data : 2020/1/8
*/

#include <iostream>
#include <string.h>
#include <math.h>
#include "tengine_c_api.h"
//#include "cpu_device.h"
#if 0
#define PRT_Q(fmt, args...) do {        \
} while (0)
#else
#define PRT_Q(fmt, args...) do {                    \
        printf("--Debug-- %d/%s() :" fmt  ,        \
             __LINE__, __func__ , ##args);        \
} while (0)
#endif
#define FLOAT_TO_REALSIZE (4)
int create_input_node(graph_t graph, const char* node_name, int c, int h, int w)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    PRT_Q("node_name = %s \n", node_name);
    tensor_t tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    PRT_Q("node = %p  tensor = %p TENSOR_TYPE_INPUT=%d \n", node, tensor ,TENSOR_TYPE_INPUT );
    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, c, h, w};

    set_tensor_shape(tensor, dims, 4);

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_conv_node(graph_t graph, const char* node_name, const char* input_name, int kernel_h,int kernel_w, int stride, int pad_h,int pad_w,
                     int in_c, int out_c, int group, int dilation ,int activation)
{
    node_t conv_node = create_graph_node(graph, node_name, "Convolution");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(input_tensor == nullptr)
    {
        std::cout << "ERRNO: " << get_tengine_errno() << "\n";
        return -1;
    }

    set_node_input_tensor(conv_node, 0, input_tensor);

    release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);

    release_graph_tensor(output_tensor);

    /* weight */

    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {out_c, in_c/ group, kernel_h, kernel_w};

    set_tensor_shape(w_tensor, w_dims, 4);

    release_graph_node(w_node);
    release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {out_c};

    set_tensor_shape(b_tensor, b_dims, 1);

    set_node_input_tensor(conv_node, 2, b_tensor);
    release_graph_node(b_node);
    release_graph_tensor(b_tensor);

    /* attr */
   // int pad1 = pad;
    set_node_attr_int(conv_node, "kernel_h", &kernel_h);
    set_node_attr_int(conv_node, "kernel_w", &kernel_w);
    set_node_attr_int(conv_node, "stride_h", &stride);
    set_node_attr_int(conv_node, "stride_w", &stride);
    set_node_attr_int(conv_node, "pad_h0", &pad_h);
    set_node_attr_int(conv_node, "pad_w0", &pad_w);
    set_node_attr_int(conv_node, "pad_h1", &pad_h);
    set_node_attr_int(conv_node, "pad_w1", &pad_w);
    set_node_attr_int(conv_node, "input_channel", &in_c);
    set_node_attr_int(conv_node, "output_channel", &out_c);
    set_node_attr_int(conv_node, "group", &group);
    set_node_attr_int(conv_node, "dilation_h", &dilation);
    set_node_attr_int(conv_node, "dilation_w", &dilation);
 //   set_node_attr_int(conv_node, "activation", &activation);

    release_graph_node(conv_node);

    return 0;
}

inline  graph_t create_conv_graph(int c, int h, int w, int kernel_h,int kernel_w, int stride, int pad_h,int pad_w, int out_c , int group, int dilation ,int activation)
{
    PRT_Q("c = %d , h = %d  , w = %d ,kernel_h  = %d ,stride  = %d, pad_h  = %d, out_c  = %d, group  = %d, dilation  = %d \n",
          c, h, w, kernel_h, stride, pad_h, out_c, group, dilation);

    graph_t graph = create_graph(nullptr, nullptr, nullptr);
    PRT_Q("graph = %p  graph = %p \n", &graph, graph );

    if(graph == nullptr)
    {
        std::cerr << "ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    const char* input_name = "data";
    const char* conv_name = "conv";

    if(create_input_node(graph, input_name, c, h, w) < 0)
    {
        std::cerr << "create input failed\n";
        return nullptr;
    }

    if(create_conv_node(graph, conv_name, input_name, kernel_h, kernel_w, stride, pad_h, pad_w, c, out_c, group, dilation, activation) < 0)
    {
        std::cerr << "create conv node failed\n";
        return nullptr;
    }

    /* set input/output node */

    const char* inputs[] = {input_name};
    const char* outputs[] = {conv_name};
    if(set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set inputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    if(set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        std::cerr << "set outputs failed: ERRNO: " << get_tengine_errno() << "\n";
        return nullptr;
    }

    return graph;
}

//size is sizeof(float)
int fill_data_increase(float  * ptr , int size ,int limit)
{
    for(unsigned int i = 0; i < size; i++)
    {
        ptr[i] = 0.0001*(i%limit);
    }
    return 0;
}
//size is sizeof(float)
int fill_data_1(float  * ptr , int size)
{
    for(unsigned int i = 0; i < size; i++)
    {
        ptr[i] = 1;
    }
    return 0;
}
int main_test(void)
{
    float * input_buf  ; float * weight_buf ; float * bias_buf  ; float * out_buf ;
    int weight_size = 0 , input_size  = 0 , out_size = 0  ,bias_size = 0  ;
    int in_c= 0,   in_h= 0 ,  in_w= 0 ;
    int kernel_h= 0 ,kernel_w= 0;
    int group= 0 , dilation= 0 , activation = 0, stride= 0 ,  pad_h= 0,  pad_w= 0 ;
    int output_c= 0 ;

    int ret = 0; int out_w = 0;int out_h = 0;
    //input_data
    // in_c = 3; in_h = 56 ; in_w = 56; kernel_h = 3;
    // stride = 1;pad_h = 1; pad_w =1; group=1 ;dilation=1 ; activation = 1 ;
    // output_c= 1 ;

    in_c = 16; in_h = 128 ; in_w = 128; kernel_h = 3; kernel_w = 3;
    stride = 1;pad_h = 1; pad_w =1; group=1 ;dilation=1 ; activation = 7 ;
    output_c= 16;

    input_size = in_c * in_h * in_w;
    weight_size = kernel_h* kernel_w* output_c * in_c ;
    bias_size = output_c;
    out_h =( in_h  +  (2 * pad_h) - kernel_h - (kernel_h-1)*(dilation-1 ))/stride + 1 ;
    out_w =( in_w  +  (2 * pad_w) - kernel_w - (kernel_w-1)*(dilation-1 ))/stride + 1 ;

    out_size =output_c * out_w * out_h;

	printf(" input ( %d x %d x %d x %d ),output ( %d x %d x %d x %d ), kernel (%d x %d), pad (%d x %d), stride (%d x %d),dilation (%d x %d)\n ",
		1,in_c,in_h,in_w,1,output_c,out_h,out_w,kernel_w,kernel_h,pad_w,pad_h,stride,stride,dilation,dilation);
    input_buf = new float[input_size];
    weight_buf = new float[weight_size];
    bias_buf = new float[bias_size];
    out_buf = new float[out_size];
    fill_data_increase(input_buf,input_size,1000);
    fill_data_increase(weight_buf,weight_size,1000);
    fill_data_1(bias_buf,bias_size);

    init_tengine();

 //   int  cpu_list[] = {1,2,3};
 //   set_online_cpu(NULL , cpu_list , 3 );
 //   set_working_cpu( cpu_list , 3 );
    set_log_level(LOG_DEBUG);

    graph_t graph =  NULL;
    tensor_t output_tensor =  NULL;
    node_t conv_node =  NULL;

    tensor_t input_tensor =  NULL;
    int buf_size = 0;
    tensor_t weight_tensor =  NULL ;
    int input_num = 0;
    float * out_ptr = NULL;

    //**************************************************************************************************************************
        graph = create_conv_graph(in_c, in_h, in_w,  kernel_h, kernel_w, stride, pad_h, pad_w, output_c, group, dilation, activation);
        PRT_Q("graph = %p  graph = %p \n", &graph, graph );
        if(graph == nullptr)
        {
            printf("Create Conv Graph (create_conv_graph) failed . \n");
            return 1;
        }
        PRT_Q("  \n");
        input_tensor = get_graph_input_tensor(graph, 0, 0);
        buf_size = get_tensor_buffer_size(input_tensor);
        if( buf_size   != input_size*FLOAT_TO_REALSIZE )
        {
            printf("Input data size check failed . buf_size = %d , input_data_size = %d \n",buf_size, input_size);
            return 1;
        }
        PRT_Q("  \n");

        set_tensor_buffer(input_tensor, (float * )input_buf, buf_size);
        release_graph_tensor(input_tensor);

        /* set weight */
        conv_node = get_graph_node(graph, "conv");
        weight_tensor = get_node_input_tensor(conv_node, 1);
        buf_size = get_tensor_buffer_size(weight_tensor);
        if( buf_size != weight_size*FLOAT_TO_REALSIZE )
        {
            printf("Input weight size check failed . buf_size = %d , input_weight_size = %d \n",buf_size, weight_size);
            return 1;
        }
        PRT_Q("  \n");

        set_tensor_buffer(weight_tensor, weight_buf, buf_size);
        PRT_Q("  \n");

        /* set bias */
        input_num = get_node_input_number(conv_node);
        PRT_Q("input_num= %d \n",input_num);

        if(input_num > 2)
        {
            tensor_t bias_tensor = get_node_input_tensor(conv_node, 2);

            buf_size = get_tensor_buffer_size(bias_tensor);
            if( buf_size != bias_size*FLOAT_TO_REALSIZE )
            {
                printf("Input bias size check failed . buf_size = %d , bias_size = %d \n",buf_size, bias_size);
            // return 1;
            }
            PRT_Q("  \n");
            set_tensor_buffer(bias_tensor, bias_buf, buf_size);
        }

        PRT_Q("  \n");

        if(prerun_graph(graph) < 0)
        {
            std::cerr << "prerun_graph failed: ERRNO: " << get_tengine_errno() << "\n";
            return 1;
        }

        dump_graph(graph);
        PRT_Q("  \n");

        if(run_graph(graph, 1) < 0)
        {
            std::cerr << "run_graph failed \n"  ;
            return 1;
        }
        PRT_Q("  \n");

        output_tensor = get_node_output_tensor(conv_node, 0);
        out_ptr = ( float * )get_tensor_buffer(output_tensor);
        buf_size = get_tensor_buffer_size(output_tensor);
 //        printf("Real : out_ptr=%p , out_ptr_d=%p \n", &out_ptr, out_ptr);
        if( buf_size != out_size*4)
        {
            printf("output_size check failed . buf_size = %d ,output_size = %d \n",buf_size , out_size);
            return 1;
        }
        PRT_Q("  \n");

        release_graph_node(conv_node);
        release_graph_tensor(output_tensor);
   //     postrun_graph(graph);
   //     destroy_graph(graph);

#if 1
    // print input and output buffer
    {
        float* indata= (float* )input_buf ;
        float* outdata=  (float* )out_buf;
        float* outdata_r=  (float* )out_ptr;
        //float* outdata = ( float * )get_tensor_buffer(output_tensor);

        printf(" --------------intput_data-------------------  \n");
        for(int ind=0 ; ind < 80 ; ind++)
        {
            printf("  %08.05f  ",indata[ind]);
            if(!((ind+1)%10))
                printf("\n");
        }
        printf(" --------------output_data-------------------  \n");
        for(int outd=0 ; outd < 80 ; outd++)
        {
            printf("  %08.05f  ",outdata[outd]);
            if(!((outd+1)%10))
                printf("\n");
        }
         printf(" --------------output_data_r-------------------  \n");
        for(int outd=0 ; outd < 80 ; outd++)
        {
            printf("  %08.05f  ",outdata_r[outd]);
            if(!((outd+1)%10))
                printf("\n");
        }
    }
#endif

    // release_tengine()   ;
    // delete[] input_buf   ;
    // delete[] weight_buf  ;
    // delete[] bias_buf    ;
    // delete[] out_buf     ;
    return 0;
}

int main(int argc, char* argv[])
{
    for (int min_i = 0 ; min_i < 1 ; min_i ++)
    {
        printf(" min_i = %d --------------------------------------------------------------------  \n",min_i);
        main_test();
    }
    return 0;
}