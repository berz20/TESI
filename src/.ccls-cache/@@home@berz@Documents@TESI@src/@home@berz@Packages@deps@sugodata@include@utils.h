#ifndef sugodata_utils_h
#define sugodata_utils_h

#include "Sugodata.h"
#include <string>

std::string approximate_to_digit(double n, int n_digits);
bool constant_errors(Sugodata &sd);

#endif // !sugodata_utils_h
