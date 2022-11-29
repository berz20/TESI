#ifndef sugodata_root_h
#define sugodata_root_h

#include "sugodata.h"

#include <TStyle.h>
#include "TH1F.h"
#include "TF1.h"
#include "TLine.h"
#include "TMath.h"
#include "TGraphErrors.h"
#include "TCanvas.h"

TGraphErrors *sugodata_to_graph_error(Sugodata &x, Sugodata &y, TCanvas *c=nullptr, std::string title="", std::string x_title="", std::string y_title="");
TH1F *sugodata_xy_to_th1f(Sugodata &x, Sugodata &y, TCanvas *c=nullptr, double bin_width=1, std::string title="", std::string x_title="", std::string y_title="");
TH1F *sugodata_frequency_th1f(Sugodata &x, TCanvas *c=nullptr, double bin_width=1, std::string title="", std::string x_title="", std::string y_title="");

#endif // !sugodata_root_h
