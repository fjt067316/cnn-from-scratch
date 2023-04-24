CC = g++
CFLAGS = -pthread -std=c++11 -I./classes
TARGET = main.o
HEADERS = model.h classes/tensor.h classes/template.h classes/activation_functions.h classes/conv.h classes/softmax.h classes/fcl.h classes/pool.h classes/utils.h classes/zeros.h classes/batch_norm.h

all: $(TARGET)

$(TARGET): main.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
