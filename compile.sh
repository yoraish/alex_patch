g++ -std=c++11  alex_patch.cpp main.cpp $(pkg-config --cflags --libs opencv) -o alex_patch_exec 
./alex_patch_exec