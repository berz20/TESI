#include "include/utils.h"
#include "include/Sugodata.h"
#include <sstream>
#include <iomanip>
#include <cmath>

#include <vector>

std::string approximate_to_digit(double n, int digit){
    std::stringstream ss;
    if(digit >= 0){
        ss << std::fixed << std::setprecision(digit) << n;
        return ss.str();
    }
    else if(digit == -1){
        ss << std::fixed << std::setprecision(0) << n;
        return ss.str();
    }

    int int_n = (int)n;
    int power_10 = std::pow(10, -digit-1);
    int rest = int_n % power_10;

    int_n -= rest;
    if(rest >= power_10 / 2.){
        int_n += power_10;
    }
    
    return std::to_string(int_n);
}

/* Return 1 if ums are constant, 2 if errors and ums are constant, 0 otherwise */
bool constant_errors(Sugodata &sd){
   double error_ref = sd[0].error;
    for(size_t i=1; i<sd.size(); i++)
        if(sd[i].error != error_ref) return false;

    return true;
}
