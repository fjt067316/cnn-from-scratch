CC = g++
CFLAGS = -pthread -std=c++11
TARGET = main.o

all: $(TARGET)

$(TARGET): main.cpp classes.h
	$(CC) $(CFLAGS) -o $(TARGET) main.cpp

clean:
	rm -f $(TARGET)
