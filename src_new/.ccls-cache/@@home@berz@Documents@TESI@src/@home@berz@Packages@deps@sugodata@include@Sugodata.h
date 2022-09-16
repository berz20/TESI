#ifndef Sugodata_h
#define Sugodata_h

#include <fstream>
#include <string>
#include <vector>

#include "Sugodatum.h"
#include "Unit.h"

class Sugodata {
  friend std::ostream &operator<<(std::ostream &os, const Sugodata &sd);

public:
  Sugodata();
  Sugodata(Unit unit);

  Sugodata(std::istream &is, std::string separator = ",");
  Sugodata(std::istream &is, Unit unit, std::string separator = ",");
  Sugodata(std::istream &is, std::istream &is2);
  Sugodata(std::istream &is, std::istream &is2, Unit unit);

  ~Sugodata();

  bool operator==(const Sugodata &other) const;
  bool operator!=(const Sugodata &other) const;

  Sugodata operator+(const Sugodata &other) const;
  Sugodata operator-(const Sugodata &other) const;
  Sugodata operator*(const Sugodata &other) const;
  Sugodata operator/(const Sugodata &other) const;

  std::vector<Sugodatum>::iterator begin();
  std::vector<Sugodatum>::const_iterator begin() const;

  std::vector<Sugodatum>::iterator end();
  std::vector<Sugodatum>::const_iterator end() const;

  size_t size() const;

  Sugodatum &operator[](const int index);
  const Sugodatum get(const int index) const;

  void setValue(const double the_value);
  void setValue(double (*f)(const double, const double));

  void setError(const double the_error);
  void setError(const double percentage, const double dgt);
  void setError(double (*f)(const double, const double));

  bool convert(Unit);
  bool use(Unit);

  bool multiple(int the_power);

  bool add(const Sugodatum sd);
  void sort(bool descending = false);

  Sugodatum &min();
  Sugodatum &max();

  double *valuesPtr() const;
  double *errorsPtr() const;

  std::shared_ptr<double> valuesSharedPtr() const;
  std::shared_ptr<double> errorsSharedPtr() const;

  double sumValues() const;
  Sugodatum mean() const;
  Sugodatum sigma(int dof = 1) const;

  /* private: */
  std::shared_ptr<Unit> unit;
  std::vector<Sugodatum> data;
};

#endif // !Sugodata_h
