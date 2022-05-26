BASE = ../../rai

include $(BASE)/config.mk

EXT    = $(shell $(PYTHON_BIN_PATH)/python3-config --extension-suffix)
OUTPUT = doorEnvironment$(EXT)

DEPEND = Core Kin Gui Geo KOMO Optim

PYBIND = 1
TORCH = 1
BULLET = 1

CXXFLAGS += $(shell $(PYTHON_BIN_PATH)/python3-config --cflags)

LIBS += -ltorch_python

OBJS = main.o doorEnvironment.o cameraObs.o

include $(BASE)/build/generic.mk
