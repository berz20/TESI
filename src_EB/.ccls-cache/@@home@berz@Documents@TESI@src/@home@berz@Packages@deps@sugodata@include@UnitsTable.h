#ifndef UnitsTable_h
#define UnitsTable_h

#include <unordered_map>
#include <string>

class Unit;

namespace units{

    enum baseunit{
        adimensional, 
        meter, 
        gram,
        candela,
        ampere,  
        kelvin,
        mol,
        second, 

        radian, degree,
        inch, foot, yard, mile, nautic_mile,
        minute, hour, day, year, hertz,
        pound, ounce,
        liter, gallon, pint,
        pascal, bar, torricelli, atmosphere, pound_square_inch,
        kilometer_hour, knot,
        galileo,
        newton, dyne,
        joule, elettronvolt, calorie,
        watt,
        coulomb,
        volt,

        tesla,

    };

    extern std::unordered_map<baseunit, const char*> table;

    Unit operator *(const baseunit &bu1, const baseunit &bu2);
    Unit operator *(const baseunit &bu1, const Unit &u2);
    Unit operator /(const baseunit &bu1, const baseunit &bu2);
    Unit operator /(const baseunit &bu1, const Unit &u2);
    Unit operator ^(const baseunit &bu1, const int power);
    Unit multiple(const baseunit &bu, const int mult);
}

#endif // !UnitsTable_h
