#include "include/Sugodata.h"

#include <algorithm>
#include <cmath>

Sugodata::Sugodata(){
    unit = nullptr;
}

Sugodata::Sugodata(Unit the_unit){
    unit = std::shared_ptr<Unit>(new Unit(the_unit));
}

Sugodata::Sugodata(std::istream &is, std::string separator){
    if(is.good()){
        std::string line;

        while(std::getline(is, line)){
            if(is.eof())
                break;

            size_t sep = line.find(separator);
            if(sep == std::string::npos){
                std::cerr << "Error: no separator in file" << std::endl;
                exit(1);
            }

            add(Sugodatum(std::stod(line.substr(0, sep)), std::stod(line.substr(sep+1))));
        }
    }
}

Sugodata::Sugodata(std::istream &is, Unit the_unit, std::string separator){
    unit = std::shared_ptr<Unit>(new Unit(the_unit));

    if(is.good()){
        std::string line;

        while(std::getline(is, line)){
            if(is.eof())
                break;

            size_t sep = line.find(separator);
            if(sep == std::string::npos){
                std::cerr << "Error: no separator in file" << std::endl;
                exit(1);
            }

            add(Sugodatum(std::stod(line.substr(0, sep)), std::stod(line.substr(sep+1)), unit));
        }
    }
}

Sugodata::Sugodata(std::istream &is, std::istream &is2){
    if(is.good() && is2.good()){
        std::string line1, line2;

        while(std::getline(is, line1) && std::getline(is2, line2)){
            if(is.eof() || is2.eof())
                break;

            data.push_back(Sugodatum(std::stod(line1), std::stod(line2)));
        }
    }
}

Sugodata::Sugodata(std::istream &is, std::istream &is2, Unit the_unit){
    unit = std::shared_ptr<Unit>(new Unit(the_unit));

    if(is.good() && is2.good()){
        std::string line1, line2;
        
        while(std::getline(is, line1) && std::getline(is2, line2)){
            if(is.eof() || is2.eof())
                break;

            data.push_back(Sugodatum(std::stod(line1), std::stod(line2), unit));
        }
    }
}

Sugodata::~Sugodata(){
}

bool Sugodata::operator ==(const Sugodata &other) const{
    if((unit == nullptr && other.unit == nullptr) || (*unit == *other.unit)){
        return data == other.data;
    }
    else if(unit->compatible(*other.unit)){
        Sugodata tmp = other;

        tmp.convert(*unit);

        return data == tmp.data;
    }
    else
        return false;
}

bool Sugodata::operator !=(const Sugodata &other) const{
    return !(*this == other);
}

Sugodata Sugodata::operator +(const Sugodata &other) const{
    if(!unit->compatible(*other.unit)){
        std::cerr << "Sugodata error: trying to sum " << *unit << " with incompatible " << *other.unit << std::endl;
        return Sugodata();
    }

    if(size() != other.size()){
        std::cerr << "Sugodata error: trying to sum sugodata of different sizes: " << size() << " and " << other.size() << std::endl; 
        return Sugodata();
    } 

    Sugodata sum(*unit);
    for(size_t i=0; i<size(); i++)
        sum.add(data.at(i) + other.get(i));
    
    
    return sum;
}

Sugodata Sugodata::operator -(const Sugodata &other) const{
    if(!unit->compatible(*other.unit)){
        std::cerr << "Sugodata error: trying to subtract " << *other.unit << " from incompatible " << *unit << std::endl;
        return Sugodata();
    }

    if(size() != other.size()){
        std::cerr << "Sugodata error: trying to subtract sugodata of different sizes: " << size() << " and " << other.size() << std::endl; 
        return Sugodata();
    } 

    Sugodata difference(*unit);
    for(size_t i=0; i<size(); i++)
        difference.add(data.at(i) - other.get(i));
    
    
    return difference;
}

Sugodata Sugodata::operator *(const Sugodata &other) const{
    if(size() != other.size()){
        std::cerr << "Sugodata error: trying to multiply sugodata of different sizes: " << size() << " and " << other.size() << std::endl; 
        return Sugodata();
    } 

    Sugodata product(*unit * *other.unit);
    for(size_t i=0; i<size(); i++)
        product.add(data.at(i) * other.get(i));
    
    
    return product;
}

Sugodata Sugodata::operator /(const Sugodata &other) const{
    if(size() != other.size()){
        std::cerr << "Sugodata error: trying to divide sugodata of different sizes: " << size() << " and " << other.size() << std::endl; 
        return Sugodata();
    } 

    Sugodata division(*unit / *other.unit);
    for(size_t i=0; i<size(); i++)
        division.add(data.at(i) / other.get(i));
    
    
    return division;
}

void Sugodata::setValue(const double the_value){
    for(Sugodatum &sd: data)
        sd.value = the_value;
}

void Sugodata::setValue(double (*f)(const double, const double)){
    for(Sugodatum &sd: data)
        sd.value = f(sd.value, sd.error);
}


void Sugodata::setError(const double the_error){
    for(Sugodatum &sd : data)
        sd.setError(the_error);
}

void Sugodata::setError(const double percentage, const double dgt){
    for(Sugodatum &sd : data)
        sd.setError(percentage, dgt);
}

void Sugodata::setError(double (*f)(const double, const double)){
    for(Sugodatum &sd: data)
        sd.error = f(sd.value, sd.error);
}

bool Sugodata::convert(Unit new_unit){
    if(unit == nullptr)
        return false;

    if(unit->components != new_unit.components)
        return false;
   
    return use(new_unit);
}

bool Sugodata::use(Unit new_unit){
    if(unit == nullptr)
        return false;
   
    double conversion_factor = unit->factor / new_unit.factor;
    
    for(Sugodatum &sd: data){
        sd.value *= conversion_factor; 
        sd.error *= conversion_factor;
    }

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

bool Sugodata::multiple(int the_power){
    if(unit == nullptr)
        return false;

    for(Sugodatum &sd: data){
        sd.value /= std::pow(10, the_power - unit->multy);
        sd.error /= std::pow(10, the_power - unit->multy);
    }
    unit->multy = the_power;

    return true;
}


std::vector<Sugodatum>::iterator Sugodata::begin(){
    return data.begin();
}

std::vector<Sugodatum>::const_iterator Sugodata::begin() const{
    return data.begin(); 
}


std::vector<Sugodatum>::iterator Sugodata::end(){
    return data.end();
}

std::vector<Sugodatum>::const_iterator Sugodata::end() const{
    return data.end(); 
}


Sugodatum& Sugodata::operator[](const int index){
    if((index > 0 && index >= (int)size()) || (index < 0 && index < -(int)size())){
        std::cerr << "index " << index << " is out of range for operator []" << std::endl;
        exit(1);
    }

    return (index >= 0)?
        data.at(index):
        data.at(size()+index);
}

const Sugodatum Sugodata::get(const int index) const{
    if((index > 0 && index >= (int)size()) || (index < 0 && index < -(int)size())){
        std::cerr << "index " << index << " is out of range for function get" << std::endl;
        exit(1);
    }

    return (index >= 0)?
        data.at(index):
        data.at(size()+index);
}


size_t Sugodata::size() const{
    return data.size();
}

bool Sugodata::add(Sugodatum sd){
    if(sd.unit == nullptr)
        sd.unit = unit;

    else if(*unit != *sd.unit)
        return false;

    else if(unit != sd.unit){
        sd.unit.reset();
        sd.unit = unit;
    }
    
    data.push_back(sd);
    return true;
}

void Sugodata::sort(bool descending){
    descending?
        std::sort(begin(), end(), [](Sugodatum &s1, Sugodatum &s2){return s1.value > s2.value;}):
        std::sort(begin(), end(), [](Sugodatum &s1, Sugodatum &s2){return s1.value < s2.value;});
}

Sugodatum& Sugodata::min(){
    if(size() == 0){
        exit(1);
        std::cerr << "asked for minimum in empty sugodata" << std::endl;
    }

    Sugodatum *the_min = &data.at(0);
    for(size_t i=1; i<size(); i++){
        if(data[i] < *the_min)
            the_min = &data[i];
    }

    return *the_min;
}

Sugodatum& Sugodata::max(){
    if(size() == 0){
        exit(1);
        std::cerr << "asked for maximum in empty sugodata" << std::endl;
    }

    Sugodatum *the_max = &data.at(0);
    for(size_t i=1; i<size(); i++){
        if(data[i] > *the_max)
            the_max = &data[i];
    }

    return *the_max;
}

double* Sugodata::valuesPtr() const{
    double *values = new double[size()];
    for(size_t i=0; i<size(); i++)
        *(values + i) = (data[i].value) ;

    return values;
}

double*  Sugodata::errorsPtr() const{
    double *errors = new double[size()];
    for(size_t i=0; i<size(); i++)
        *(errors + i) = (data[i].error) ;

    return errors;
}

std::shared_ptr<double> Sugodata::valuesSharedPtr() const{
    std::shared_ptr<double> values(new double[size()]);
    for(size_t i=0; i<size(); i++)
        *(values.get() + i) = (data[i].value) ;

    return values;
}

std::shared_ptr<double> Sugodata::errorsSharedPtr() const{
    std::shared_ptr<double> errors(new double[size()]);
    for(size_t i=0; i<size(); i++)
        *(errors.get() + i) = (data[i].error) ;

    return errors;
}

double Sugodata::sumValues() const{
    double sum=0;
    for(const Sugodatum &s : data)
        sum += s.value;

    return sum;
}
    
Sugodatum Sugodata::mean() const{    
    return Sugodatum(sumValues()/size(), sigma().value / std::sqrt(size()), unit);
}

Sugodatum Sugodata::sigma(int dof) const{
    int N = size();
    if(N == 1){
        return Sugodatum(data.at(0).error, data.at(0).error, unit);
    }

    double variance=0, the_mean = sumValues()/size();
     
    for (const Sugodatum &x : data)
        variance += (x.value - the_mean) * (x.value - the_mean);
   
    double sigma = std::sqrt(variance / (N - dof));

    return Sugodatum(sigma, sigma/std::sqrt((dof+1)*(N - 1)), unit);
}

std::ostream& operator <<(std::ostream &os, const Sugodata &sd){
    for(Sugodatum s : sd.data)
        os << s << std::endl;

    return os;
}
