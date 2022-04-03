#include <string>

bool readFile(std::string name, float** pics, int& width, int& heigh);
void saveFile(std::string name, float* pics, int width, int heigh, int maxV);