#include "include/Unit.h"
#include "include/UnitsTable.h"
#include <iostream>
#include <map>
#include <cstring>

Unit::Unit(){
    name = "";
    factor = 1;
    power = 0;
    multy = 0;
}
    
Unit::Unit(units::baseunit bu, int the_multy){
    std::string description = units::table[bu];

    size_t column = description.find(":");
    name = description.substr(0, column);

    size_t star = description.find("*", column+1);
    factor = std::stod(description.substr(column+1, star-column));

    size_t bar = description.find("|", star+1);
    power = std::stoi(description.substr(star+1, bar-star));

    size_t i = bar+1; 
    while (i < description.size()){

        size_t multy = description.find("^", i);
        std::string tmp_unit = description.substr(i, multy-i);

        size_t ref = multy + 1;
        
        size_t comma = description.find(",", ref);
        std::string tmp_multy = description.substr(ref, comma-ref);

        components[tmp_unit] = stoi(tmp_multy);

        if(comma == std::string::npos)
            break;
        i = comma + 1;
    }

    multy = the_multy;
}

Unit::Unit(std::string the_name, double the_factor, int the_power, std::string components_str, int the_multy){
    name = the_name;
    factor = the_factor;
    power = the_power;
    multy = the_multy;

    size_t i=0;
    while (i < components_str.size()){
        size_t multy = components_str.find("^", i);
        std::string tmp_unit = components_str.substr(i, multy-i);

        size_t ref = multy + 1;
        
        size_t comma = components_str.find(",", ref);
        std::string tmp_multy = components_str.substr(ref, comma-ref);

        components[tmp_unit] = stoi(tmp_multy);

        if(comma == std::string::npos)
            break;
        i = comma + 1;
    }
}

Unit::~Unit(){}

bool Unit::operator == (const Unit &other) const{
    return name == other.name && factor == other.factor && power == other.power && components == other.components;
}

bool Unit::operator != (const Unit &other) const{
    return !(*this == other);
}

Unit Unit::operator *(const Unit &other) const{
    Unit the_product = *this;

    the_product *= other;

    return the_product;
}

Unit Unit::operator /(const Unit &other) const{
    Unit the_division = *this;

    the_division /= other;

    return the_division;
}

void Unit::operator *=(const Unit &other){
    for(auto &comp : other.components){
        if(comp.second != 0){
            components[comp.first] += comp.second;
            if(components[comp.first] == 0)
                components.erase(comp.first);


            if(other.name.find("*") != std::string::npos){
                if(name != " ")
                    name += "*";
                name += comp.first;

                if(comp.second != 1)
                    name += "^" + std::to_string(comp.second);
            }
        }
    }

    if(other.name.find("*") == std::string::npos){
        if(name != " ")
            name += "*";
       
        size_t at;
        if((at = other.name.find("^")) == std::string::npos)
           name += other.name + "^-1";
        else
           name += other.name + "^" + std::to_string(std::stoi(other.name.substr(at+1)) - 1);
    }

    if(other.name.find("*") == std::string::npos){
        if(name != " ")
            name += "*";
       
        size_t at;
        if((at = other.name.find("^")) == std::string::npos)
           name += other.name;
        else
           name += other.name + "^" + std::to_string(std::stoi(other.name.substr(at+1)) + 1);
    }

    factor = factor * other.factor;
    power += other.power;
    multy += other.multy;
}

void Unit::operator /=(const Unit &other){
    for(auto &comp : other.components){
        if(comp.second != 0){
            components[comp.first] -= comp.second;
            if(components[comp.first] == 0)
                components.erase(comp.first);
            
            if(other.name.find("*") != std::string::npos){
                if(name != " ")
                    name += "*";
                name += comp.first;

                if(comp.second != -1)
                    name += "^" + std::to_string(-comp.second);
            }
        }
    }

    if(other.name == name)
        name = "";

    else if(other.name.find("*") == std::string::npos){
        if(name != " ")
            name += "*";
       
        size_t at;
        if((at = other.name.find("^")) == std::string::npos)
           name += other.name + "^-1";
        else
           name += other.name + "^" + std::to_string(std::stoi(other.name.substr(at+1)) - 1);
    }

    factor = factor / other.factor;
    power -= other.power;
    multy -= other.multy;
}

Unit Unit::operator ^(const int exponent) const{
    Unit tmp;
    if(exponent == 0)
        return tmp;

    if(exponent == 1)
        return *this;

    tmp.name = name + "^" + std::to_string(exponent);
    tmp.factor = 1;
    tmp.power = 0;
    tmp.multy = 0;

    if(exponent > 0){
        for(int i=0; i<exponent; i++){
            tmp.factor *= factor; 
            tmp.power += power; 
            tmp.multy += multy; 

            for(auto &comp: components){
                tmp.components[comp.first] += comp.second;
            }
        }
    }

    else if(exponent < 0){
        for(int i=0; i>exponent; i--){
            tmp.factor /= factor; 
            tmp.power -= power; 
            tmp.multy -= multy; 

            for(auto &comp: components){
                tmp.components[comp.first] -= comp.second;
            }
        }
    }

    return tmp;
}

const std::string Unit::dimensionString() const{
    std::string dim;
    size_t n=0;
    for(auto &comp : components){
        dim += comp.first;
        if(comp.second != 1)
            dim += "^" + std::to_string(comp.second);

        if(++n < components.size())
            dim += "*";
    }
    return dim;
}
    
bool Unit::compatible(const Unit &other) const{
    return components == other.components;
}

void Unit::print(int mode, std::ostream &os) const{
    os << " ";
    if(mode == 0 && name.find("*") == std::string::npos){
        size_t pow_char;
        int power = multy;
        if((pow_char = name.find("^")) != std::string::npos){
            int the_power = std::stoi(name.substr(pow_char + 1));
            power /= the_power;
        }
        switch(power){
            case -24:
                os << "y";
                break;
            case -21:
                os << "z";
                break;
            case -18:
                os << "a";
                break;
            case -15:
                os << "f";
                break;
            case -12:
                os << "p";
                break;
            case -9:
                os << "n";
                break;
            case -6:
                os << "u";
                break;
            case -3:
                os << "m";
                break;
            case -2:
                os << "c";
                break;
            case -1:
                os << "d";
                break;
            case 0:
                break;
            case 1:
                os << "Da";
                break;
            case 2:
                os << "h";
                break;
            case 3:
                os << "k";
                break;
            case 6:
                os << "M";
                break;
            case 9:
                os << "G";
                break;
            case 12:
                os << "T";
                break;
            case 15:
                os << "P";
                break;
            case 18:
                os << "E";
                break;
            case 21:
                os << "Z";
                break;
            case 24:
                os << "Y";
                break;
            default:
                os << "* 10^" << multy << " ";
                break;
        }
    } 
    else{
        if(multy != 0)
            os << "* 10^" << multy << " ";
    }

    os << name;
}

std::ostream &operator <<(std::ostream &os, const Unit &u){
    u.print(0, os);
    return os;
}
