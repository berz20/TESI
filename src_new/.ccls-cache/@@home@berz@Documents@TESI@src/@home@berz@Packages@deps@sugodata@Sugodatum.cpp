#include "include/Sugodatum.h"
#include "include/utils.h"
#include <cmath>
#include <sstream>
#include <iomanip>

Sugodatum::Sugodatum(){
    value = 0;
    error = 0;
    unit = nullptr;
}

Sugodatum::Sugodatum(double v, double e){
    value = v; 
    error = std::abs(e);
    unit = nullptr;
}

Sugodatum::Sugodatum(double v, double e, Unit the_unit){
    value = v;
    error = std::abs(e);
    unit = std::shared_ptr<Unit>(new Unit(the_unit));
}

Sugodatum::Sugodatum(double v, double e, std::shared_ptr<Unit> the_unit){
    value = v;
    error = std::abs(e);
    unit = the_unit;
}

Sugodatum::~Sugodatum(){
}

const std::unordered_map<std::string, int>* Sugodatum::dimension() const{
    if(unit == nullptr)
        return nullptr;
    
    return &unit->components;
}

const std::string Sugodatum::dimensionString() const{
    if(unit == nullptr)
        return nullptr;

    return unit->dimensionString();
}

bool Sugodatum::compatible(const Sugodatum &other) const{
    return this->unit->compatible(*other.unit);
}

bool Sugodatum::multiple(int the_power){
    if(unit == nullptr)
        return false;

    value /= std::pow(10, the_power - unit->multy);
    error /= std::pow(10, the_power - unit->multy);
    unit->multy = the_power;

    return true;
}

bool Sugodatum::operator ==(const Sugodatum &other) const{
    if((unit == nullptr && other.unit == nullptr) || (*unit == *other.unit)){
        return value == other.value && error == other.error;
    }
    else if(unit->compatible(*other.unit)){
        Sugodatum tmp = other;

        tmp.convert(*unit);

        return value == tmp.value && error == tmp.error;
    }
    else
        return false;
}

bool Sugodatum::operator !=(const Sugodatum &other) const{
    return !(*this == other);
}


bool Sugodatum::operator <(const Sugodatum &other) const{
    return value < other.value;
}

bool Sugodatum::operator >(const Sugodatum &other) const{
    return value > other.value;
}

bool Sugodatum::operator <=(const Sugodatum &other) const{
    return value <= other.value;
}

bool Sugodatum::operator >=(const Sugodatum &other) const{
    return value <= other.value;
}

void Sugodatum::operator *=(const double factor){
    value *= factor; 
    error *= factor; 
}

void Sugodatum::operator /=(const double factor){
    value /= factor; 
    error /= factor; 
}

Sugodatum Sugodatum::operator *(const double factor) const{
    Sugodatum the_product(*this);

    the_product *= factor;

    return the_product;
}

Sugodatum Sugodatum::operator /(const double factor) const{
    Sugodatum the_division(*this);

    the_division /= factor;

    return the_division;
}

Sugodatum Sugodatum::operator +(const Sugodatum &other) const{
    if(!unit->compatible(*other.unit)){
        std::cerr << "Sugodata error: trying to sum " << *unit << " with incompatible " << *other.unit << std::endl;
        return Sugodatum();
    }
    
    return Sugodatum(value + other.value, std::sqrt(error*error + other.error*other.error), *unit);
}

Sugodatum Sugodatum::operator -(const Sugodatum &other) const{
    if(!unit->compatible(*other.unit)){
        std::cerr << "Sugodata error: trying to subtract " << *other.unit << " from incompatible " << *unit << std::endl;
        return Sugodatum();
    }

    return Sugodatum(value - other.value, std::sqrt(error*error + other.error*other.error), *unit);
}

Sugodatum Sugodatum::operator *(const Sugodatum &other) const{
    return Sugodatum(value * other.value, std::sqrt(error*error * other.value*other.value + other.error*other.error * value*value), *unit * *other.unit);
}

Sugodatum Sugodatum::operator /(const Sugodatum &other) const{
    return Sugodatum(value / other.value, std::sqrt(error*error + other.error*other.error * value*value / (other.value*other.value)) / std::abs(other.value), *unit / *other.unit);
}

void Sugodatum::setValue(const double the_value){
    value = the_value;
}

void Sugodatum::setValue(double (*f)(const double, const double)){
    value = f(value, error);
}

void Sugodatum::setError(const double the_error){
    error = the_error;
}

void Sugodatum::setError(const double percentage, const double dgt){
    error = percentage * value + dgt;
}

void Sugodatum::setError(double (*f)(const double, const double)){
    error = f(value, error);
}

bool Sugodatum::convert(Unit new_unit){
    if(unit == nullptr)
        return false;

    if(unit->components != new_unit.components)
        return false;
   
    return use(new_unit);
}

bool Sugodatum::use(Unit new_unit){
    if(unit == nullptr)
        return false;
   
    double conversion_factor = unit->factor / new_unit.factor;
    value *= conversion_factor; 
    error *= conversion_factor;

    int the_multy = new_unit.multy;
    new_unit.multy = (unit->power + unit->multy - new_unit.power);

    if(unit->components != new_unit.components){
        std::unordered_map<std::string, int> difference;
        for(std::pair<std::string, int> comp : unit->components)
            difference[comp.first] = comp.second - new_unit.components[comp.first];  

        for(std::pair<std::string, int> comp: difference){
            if(comp.second != 0){
                new_unit.name += "*" + comp.first;
                if(comp.second != 1)
                    new_unit.name += "^" + std::to_string(comp.second);
            }
        }

        new_unit.components = unit->components;
    }

    *unit = Unit(new_unit);

    if(the_multy != 0)
        multiple(the_multy);

    return true;
}

double Sugodatum::relativeError() const{
    return value/error;
}

double Sugodatum::gaussTest(Sugodatum &reference) const{
    return (reference.value - value) / std::sqrt(error * error + reference.error * reference.error);
}

size_t Sugodatum::firstSignificantDigit() const{
    if(error == 0) return 0;
    
    int l = -std::floor(std::log10(error));
    return (error > 1)? l - 1: l;
}

void Sugodatum::printToSignificantDigits(int mode, int unit_mode, std::string separator, std::ostream &os) const{
    if(error == 0){
        os << value;

        if(unit_mode != -1 && unit != nullptr)
            unit->print(unit_mode, os);

        return;
    }

    int first_significant = firstSignificantDigit();

    int n_digits = mode;
    if(mode <= 0){
        n_digits = 1;

        std::stringstream ss;
        ss << std::fixed;
        error < 1 ?
            ss << std::setprecision(first_significant + 1):
            ss << std::setprecision(1);

        ss << error;
        std::string x_str = ss.str();

        int dot = x_str.find('.');
        
        if(dot + first_significant != (int)x_str.find_last_not_of("0") && x_str.at(dot + first_significant) == '1'){
            int check = dot + first_significant + 1;
            if(x_str.at(dot + first_significant + 1) == '.')
                check++;

            if(x_str.at(check) < '5')
                n_digits=2;
        } 
    }

    if(mode == -1)
        os << approximate_to_digit(value, first_significant + n_digits - 1);
    else if (mode == -2)
        os << approximate_to_digit(error, first_significant + n_digits - 1);
    else{
        os << approximate_to_digit(value, first_significant + n_digits - 1)
           << separator
           << approximate_to_digit(error, first_significant + n_digits - 1);
    }
    
    if(unit_mode != -1 && unit != nullptr)
        unit->print(unit_mode, os);
}
    
std::ostream &operator <<(std::ostream &os, const Sugodatum &sd){
    sd.printToSignificantDigits(0, 0, " +/- ", os);
    return os;
}
