# Toolchain
CXX := g++

# OpenCV vendorizado
OPENCV_INCLUDE := third_party/opencv/include
OPENCV_LIB_DIR := third_party/opencv/x64/mingw/lib
OPENCV_BIN_DIR := third_party/opencv/x64/mingw/bin

# Flags
CXXFLAGS := -Wall -Wextra -std=c++17 -Iinclude -isystem $(OPENCV_INCLUDE) -MMD -MP
LDFLAGS  := -L$(OPENCV_LIB_DIR) -Wl,--start-group
OPENCV_LIBS := \
  -lopencv_objdetect455 \
  -lopencv_imgcodecs455 \
  -lopencv_imgproc455 \
  -lopencv_highgui455 \
  -lopencv_core455 \
  -Wl,--end-group
# (Opcional) reduzir dependências do runtime do GCC:
# LDLIBS_EXTRA := -static-libstdc++ -static-libgcc

# Estrutura
SRCDIR := src
OBJDIR := obj
BINDIR := bin

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPS    := $(OBJECTS:.o=.d)

TARGET := $(BINDIR)/MyCppProject.exe

.PHONY: all clean run copy-dlls

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS) $(OPENCV_LIBS) $(LDLIBS_EXTRA)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

# Executar ajustando PATH para localizar as DLLs do OpenCV
run: all
	PATH="$(CURDIR)/$(OPENCV_BIN_DIR):$$PATH" ./$(TARGET)

# Opcional: copiar DLLs para bin/ após o build
copy-dlls:
	mkdir -p $(BINDIR)
	cp -u $(OPENCV_BIN_DIR)/*.dll $(BINDIR)/

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Regras de dependência (recompila ao mudar headers)
-include $(DEPS)
