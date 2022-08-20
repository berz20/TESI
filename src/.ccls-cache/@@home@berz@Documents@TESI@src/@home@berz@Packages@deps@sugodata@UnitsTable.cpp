#include "include/UnitsTable.h"
#include <string>
#include <iostream>
#include <unordered_map>

#include "include/Unit.h"

namespace units{
    Unit operator *(const baseunit &bu1, const baseunit &bu2){
        return Unit(bu1) * Unit(bu2);
    }

    Unit operator *(const baseunit &bu1, const Unit &u2){
        return Unit(bu1) * u2;
    }

    Unit operator /(const baseunit &bu1, const baseunit &bu2){
        return Unit(bu1) / Unit(bu2);
    }

    Unit operator /(const baseunit &bu1, const Unit &u2){
        return Unit(bu1) / u2;
    }

    Unit operator ^(const baseunit &bu1, const int power){
        return Unit(bu1)^power;
    }

    Unit multiple(const baseunit &bu, const int mult){
        return Unit(bu, mult);
    }

    std::unordered_map<baseunit, const char*> table{
        {adimensional, " :1*0|"},
        {meter, "m:1*0|m^1"},
        {gram, "g:1*0|g^1"},
        {candela, "cd:1*0|cd^1"},
        {ampere, "A:1*0|A^1"},
        {kelvin, "K:1*0|K^1"},
        {mol, "mol:1*0|mol^1"},
        {second, "s:1*0|s^1"},

        {radian, "rad:1*0|rad^1"},
        {degree, "deg:3.490658503988659*-2|rad^1"},


        {inch, "in:2.54*-2|m^1"},
        {foot, "ft:3.048*-1|m^1"},
        {yard, "yd:9.144*-1|m^1"},
        {mile, "mi:1.609344*3|m^1"},
        {nautic_mile, "naut mi:1.853248*3|m^1"},

        {minute, "min:6*1|s^1"},
        {hour, "h:3.6*3|s^1"},
        {day, "day:8.6400*4|s^1"},
        {year, "yr:3.1556736*7|s^1"},
        {hertz, "Hz:1*0|s^-1"},
     
        {pound, "lb:4.53592370*2|g^1"},
        {ounce, "oz:2.8349523*1|g^1"},

        {liter, "l:1*-3|m^3"},
        {gallon, "gal:3.785412*-3|m^3"},
        {pint, "pt:4.73176*-4|m^3"},
        
        {pascal, "Pa:1*3|g^1,m^-1,s^-2"},
        {bar, "bar:1*8|g^1,m^-1,s^-2"},
        {torricelli, "torr:1.333*5|g^1,m^-1,s^-2"},
        {atmosphere, "atm:1.01325*8|g^1,m^-1,s^-2"},
        {pound_square_inch, "psi:6.895*6|g^1,m^-1,s^-2"},

        {kilometer_hour, "km/h:3.6*0|m^1,s^-1"},
        {knot, "knot:0.5144*0|m^1,s^-1"},

        {galileo, "gal:1*-2|m^1,s^-2"},
 
        {newton, "N:1*3|g^1,m^1,s^-2"},
        {dyne, "dyne:1*-1|g^1,m^1,s^-2"},

        {joule, "J:1*3|g^1,m^2,s^-2"},
        {elettronvolt, "eV:1.602176634*-16|g^1,m^2,s^-2"},
        {calorie, "cal:4.1868*3|g^1,m^2,s^-2"},

        {watt, "W:1*3|g^1,m^2,s^-3"},

        {coulomb, "C:1*0|A^1,s^1"},
        {volt, "V:1*3|g^1,m^2,s^-1,A^1"},
       
        {tesla, "T:1*3|g^1,s^-2,A^-1"},
    };
}

