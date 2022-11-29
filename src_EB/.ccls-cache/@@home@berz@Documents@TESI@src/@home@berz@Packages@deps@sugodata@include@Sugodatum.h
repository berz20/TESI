#ifndef Sugodatum_h
#define Sugodatum_h

#include <string>
#include <memory>
#include <optional>

#include "Unit.h"
#include "UnitsTable.h"

class Sugodatum{

friend std::ostream &operator <<(std::ostream &os, const Sugodatum &sd);

public:
    Sugodatum();
    Sugodatum(double v, double e);
    Sugodatum(double v, double e, Unit the_unit);
    Sugodatum(double v, double e, std::shared_ptr<Unit> the_unit);
    ~Sugodatum();

    const std::unordered_map<std::string, int>* dimension() const;
    const std::string dimensionString() const;
    bool compatible(const Sugodatum &other) const;

    bool multiple(int the_power);

    bool operator ==(const Sugodatum &other) const;
    bool operator !=(const Sugodatum &other) const;

    bool operator <(const Sugodatum &other) const;
    bool operator >(const Sugodatum &other) const;
    bool operator <=(const Sugodatum &other) const;
    bool operator >=(const Sugodatum &other) const;

    void operator *=(const double);
    void operator /=(const double);

    Sugodatum operator *(const double) const;
    Sugodatum operator /(const double) const;

    Sugodatum operator +(const Sugodatum &other) const;
    Sugodatum operator -(const Sugodatum &other) const;

    Sugodatum operator *(const Sugodatum &other) const;
    Sugodatum operator /(const Sugodatum &other) const;

    void setValue(const double the_value);
    void setValue(double (*f)(const double, const double));

    void setError(const double the_error);
    void setError(const double percentage, const double dgt);
    void setError(double (*f)(const double, const double));

    bool convert(Unit);
    bool use(Unit);

    double relativeError() const;
    double gaussTest(Sugodatum &reference) const;

    size_t firstSignificantDigit() const;
    void printToSignificantDigits(int mode=1, int unit_mode=0, std::string separator=" +/- ", std::ostream &os=std::cout) const;

/* private: */
    double value;  
    double error;  
    std::shared_ptr<Unit> unit;
};


#endif // !Sugodatum_h
