CXX := clang++
SHO := oscillator.cpp
NLP := nonlinearPendulum.cpp
LORENZ := lorenz.cpp
CHUA := chua.cpp
STOKES := stokesDrag.cpp
VERLET := verlet.cpp

MAIN := $(CHUA)

BIN := bin
SRC := ./src
BUILDDIR := ./build
IR := ./ir
SRCS := $(wildcard $(SRC)/*.cpp) 
HEADERS := $(wildcard $(SRC)/*.hpp) 
HEADERS += $(wildcard $(SRC)/*.h) 
IRS :=$(patsubst %.cpp,%.ll,$(SRCS))
PROFGENBIN := profgenbin
PROFUSEBIN := profusebin

STD := -std=gnu++2b -pipe -g
LLVM := -c #-fPIE # -S -emit-llvm
WARN := -Wall -Wno-unused-function

LD := -L./ -L/usr/lib -lm -pthread -lc++ -fuse-ld=lld -lboost_iostreams -lboost_system -lboost_filesystem -lquadmath -lgmpxx -lgmp -lmpfr
INC := -I/usr/include -Isciplot/ -I/usr/include/boost -I/usr/include/boost/filesystem -I/usr/include/boost/system -I/usr/include/boost/iostreams
INFO := #--analyze
OPT := -O3 -mtune=native -march=native -flto
#OPT += -fno-signed-zeros -fno-trapping-math -fassociative-math -freciprocal-math
#OPT += -flto -fvectorize -fstrict-enums
#OPT += -fno-stack-protector -fno-stack-clash-protection
#OPT += -finline-functions -fasm -fslp-vectorize -fopenmp=libomp
#OPT += -foptimize-sibling-calls 
#OPT += -felide-constructors -fpack-struct=16 
#OPT += -faligned-allocation -fsplit-stack -fopenmp-target-new-runtime 
#OPT += -fopenmp-extensions -foffload-lto -fnew-infallible
#OPT += -fgnu-keywords
#OPT += -mno-seses -fvirtual-function-elimination
#OPT += -fwhole-program-vtables -fstrict-vtable-pointers
#OPT += -funroll-loops # -freroll-loops 
#OPT += -fjump-tables -ffinite-loops #-ffinite-math-only 
#OPT += -frewrite-imports -frewrite-includes 
#OPT += -mllvm -polly -mllvm -polly-vectorizer=stripmine -mllvm -polly-position=before-vectorizer
#OPT += -mllvm -polly-parallel -lgomp #-mllvm -polly-isl-arg=-schedule-algorithm=featurier
#OPT += -mllvm -inline-threshold=100
#OPT += -mllvm -polly-omp-backend=LLVM -mllvm -polly-num-threads=16
#OPT += -mllvm -polly-scheduling=dynamic -mllvm -polly-scheduling-chunksize=64
#OPT += -fnt-store -fremap-arrays -fitodcalls -finline-aggressive
#OPT += -falign-loops=64 -fcxx-modules -frename-registers
#OPT += -ffunction-sections -fdata-sections

CLANGGENPROF := -g -fprofile-instr-generate=./profilestuff/lastrun.raw -fcoverage-mapping
CLANGUSEPROF := -fprofile-instr-use=./profilestuff/merged.out

CXXFLAGS := $(STD) $(INFO) $(WARN) $(OPT) $(INC) $(LD)
#LDFLAGS := $(LD)
#all:
#	$(CXX) $(CXXFLAGS) -o $(BIN) *.cpp


all: 
	$(CXX) -o $(BIN) $(CXXFLAGS) $(MAIN)

clean:
	rm -f bin
	rm -f $(PROFGENBIN)
	rm -f profilestuff/lastrun.raw

cleanprof:
	rm -f bin
	rm -f $(PROFGENBIN)
	rm -f profilestuff/merged.out
	rm -f profilestuff/lastrun.raw


run:
	./bin


profile:
	$(CXX) -g $(CXXFLAGS) $(CLANGGENPROF) -o $(PROFGENBIN) $(MAIN)
	./$(PROFGENBIN)
	llvm-profdata merge -output=./profilestuff/merged.out -instr ./profilestuff/lastrun.raw

useprofile:
	$(CXX) $(CXXFLAGS) $(CLANGUSEPROF) -o $(PROFUSEBIN) $(MAIN)




