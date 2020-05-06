TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = $(shell /usr/local/anaconda3/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = $(shell /home/wj/anaconda3_cpu/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_INC = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow/include"
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = $(shell /usr/local/anaconda3/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = $(shell /home/wj/anaconda3_cpu/bin/python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#TF_LIB = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow"
USE_CUDA := "YES"
#all: build
#all: clean build
#g++ -std=c++14 -shared -g -o $@ $^ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -L${TF_LIB} -ltensorflow_framework 
#CXXFLAGS = -I ${TF_INC} -fPIC -L${TF_LIB} -ltensorflow_framework -D_GLIBCXX_USE_CXX11_ABI
CXX11_ABI=0
OPT = -O3
#OPT = -g
CUDA_OPT = -O0
#UDA_OPT = -g
CXXFLAGS = -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared ${OPT} -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -fPIC -L${TF_LIB} -ltensorflow_framework `pkg-config --cflags opencv` `pkg-config --libs opencv`
#CUDAFLAGS = -std=c++14 -g -shared -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA --expt-relaxed-constexpr -Xcompiler="-fPIC"
#CUDAFLAGS = -std=c++14 -g -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA --expt-relaxed-constexpr -Xcompiler="-fPIC" -arch=sm_30
CUDAFLAGS = -std=c++14 -I $(TF_INC) ${CUDA_OPT} -I /usr/local -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -D GOOGLE_CUDA=1 --expt-relaxed-constexpr -Xcompiler="-fPIC" -gencode arch=compute_60,code=compute_60 -x cu 
#-lcudadevrt

build: wtfop.so

#wtfop.so: wtfop.cc bboxes.cc wtoolkit.cc nlp.cc dentalfilm.cc medical.cc
ifdef USE_CUDA
#targets = mask.o wtfop.o bboxes.o wtoolkit.o nlp.o medical.o gn.o anchor_bboxes.o bboxes_encode_decode.o nms.o bboxes_encode_decode_gpu.o center_net_encode_decode.o opencvadapter.o matcher.o matcher_gpu.o ocr_ops.o deform_conv.cu.o deform_conv.o
targets = mask.o wtfop.o bboxes.o wtoolkit.o nlp.o medical.o gn.o anchor_bboxes.o bboxes_encode_decode.o nms.o bboxes_encode_decode_gpu.o center_net_encode_decode.o opencvadapter.o matcher.o matcher_gpu.o ocr_ops.o nms_gpu.o link.o
CXXFLAGS += -DGOOGLE_CUDA -DEIGEN_USE_GPU -DEIGEN_USE_THREADS
CXXFLAGS += -I/usr/local/cuda-9.0/include 
else
targets = mask.o wtfop.o bboxes.o wtoolkit.o nlp.o medical.o gn.o anchor_bboxes.o bboxes_encode_decode.o nms.o center_net_encode_decode.o opencvadapter.o matcher.o ocr_ops.o
endif

bboxes_encode_decode_gpu.o:bboxes_encode_decode_gpu.cu
	nvcc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${CUDAFLAGS}

matcher_gpu.o:matcher_gpu.cu
	nvcc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${CUDAFLAGS}

deform_conv.cu.o:thirdparty/deformable-conv/lib/deform_conv.cu.cc
	nvcc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 ${CUDAFLAGS} --expt-relaxed-constexpr -o $@

nms_gpu.o:nms_gpu.cu
	#nvcc -arch=sm_35 -dc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 -rdc=true ${CUDAFLAGS} --expt-relaxed-constexpr -o $@
	nvcc -dc $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 -rdc=true ${CUDAFLAGS} --expt-relaxed-constexpr -o $@

link.o:nms_gpu.o
	#nvcc -arch=sm_35 -dc -c $^ -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 -rdc=true ${CUDAFLAGS} --expt-relaxed-constexpr -o $@
	nvcc -dlink $^ -gencode arch=compute_60,code=compute_60  -I/usr/local/cuda-9.0/include -L/usr/local/cuda-9.0/lib64 -Xcompiler="-fPIC" -o $@

deform_conv.o:thirdparty/deformable-conv/lib/deform_conv.cc
	g++ -c $^ ${CXXFLAGS} -o $@

wtfop.so:${targets}
	echo $CXXFLAGS
	echo $^
	g++ -std=c++14 -Wno-ignored-attributes -Wno-unused-result -shared ${OPT} -o $@ $^ -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=${CXX11_ABI} -fPIC -L${TF_LIB} -ltensorflow_framework  -L/usr/local/cuda-9.0/lib64 -lcudart -lcudadevrt `pkg-config --cflags opencv` `pkg-config --libs opencv` 

clean:
	rm -f *.o *.so *.pyc *.npy
