#TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_INC = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow/include"
#TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_LIB = "/home/vghost/.local/lib/python3.6/site-packages/tensorflow"
all: clean build
#g++ -std=c++14 -shared -g -o $@ $^ -I $(TF_INC) -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -L${TF_LIB} -ltensorflow_framework 

build: wtfop.so

wtfop.so: wtfop.cc bboxes.cc wtoolkit.cc nlp.cc dentalfilm.cc
	echo $^
	g++ -std=c++14 -Wno-unused-result -shared -g -o $@ $^ -I $(TF_INC) -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -L${TF_LIB} -ltensorflow_framework 

clean:
	rm -f *.o *.so *.pyc *.npy
