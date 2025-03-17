# Makefile
# 简单版示例

NVCC        := nvcc
NVCC_FLAGS  := -shared -Xcompiler -fPIC
LD_LIBS     := -lcusolver -lcublas
TARGET      := libsvd_quant_multi.so
SRC         := libsvd_quant_multi.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LD_LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
