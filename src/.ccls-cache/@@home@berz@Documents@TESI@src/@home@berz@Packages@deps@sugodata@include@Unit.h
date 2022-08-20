#ifndef Unit_h
#define Unit_h

#include <string>
#include <iostream>
#include <unordered_map>
#include <map>

#include "UnitsTable.h"

class Unit{
public:
    Unit();
    Unit(units::baseunit, int mult=0);
    Unit(std::string the_name, double the_factor, int the_power, std::string components_str, int the_multy=0);
    ~Unit();

    bool operator == (const Unit &other) const;
    bool operator != (const Unit &other) const;

    Unit operator *(const Unit &other) const;
    Unit operator /(const Unit &other) const;

    void operator *=(const Unit &other);
    void operator /=(const Unit &other);
    
    Unit operator ^(const int) const;

    void print(int mode, std::ostream &os) const;

    const std::string dimensionString() const;

    bool compatible(const Unit &other) const;

    std::string name;
    double factor;
    int power;
    std::unordered_map<std::string, int> components;

    int multy;
};

std::ostream &operator <<(std::ostream &os, const Unit &u);

#endif // !Unit_h
