#include <fstream>
#include <iostream>
#include <math.h>
#include <vector>

#include <TCanvas.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TH1F.h>
#include <TLine.h>
#include <TMath.h>
#include <TMultiGraph.h>
#include <TPaveStats.h>
#include <TStyle.h>

#include <../../home/berz/Packages/deps/rootdeps.h>

#define output "../OUTDIR/"
const double H_PLANCK = 6.626070154e-34;
const double GYR_RATIO_DIAMOND = 0.1760862682e-6;
using namespace std;

double err_B(const double value, const double error) { return 0.011 * value; }

double err_Split(const double value, const double error) {
  return 0.011 * value;
}

Sugodata B_exp(Sugodata split) {
  Sugodata B_e(multiple(units::tesla, -3));

  // for (auto i : split.data) {
  //   B_e.add(i * PI / (GYROMAGNETIC_RATIO_DIAMOND));
  // }
  for (int i = 0; i < split.size(); i++) {
    B_e[i].value = split[i].value * M_PI / (GYR_RATIO_DIAMOND);
  }
  return B_e;
}

int splitting_zeeman(const char *file) {
  Sugodata B(multiple(units::tesla, -3)), Split(multiple(units::hertz, 9));

  sugodata_parse_file_data(file, ",", B, Split);

  Split.setError(err_Split); /* percent + dgt (da manuale) */
  B.setError(err_B);
  Sugodata B_e(multiple(units::tesla, -3));
  B_e = B_exp(Split);

  // if (B_e.add(Split.data[0] * PI / (GYROMAGNETIC_RATIO_DIAMOND)))
  //   cout << "yes" << endl;
  // else
  //   cout << "no" << endl;

  std::cout << "\033[1;35mB\033[0m" << std::endl;
  std::cout << B << std::endl;

  std::cout << "\033[1;35mSplitting\033[0m " << std::endl;
  std::cout << Split << std::endl;

  std::cout << "\033[1;35mB_e\033[0m" << std::endl;
  std::cout << B_e << std::endl;

  TCanvas *c = new TCanvas();
  TMultiGraph *multi = new TMultiGraph();

  TGraphErrors *graph_sonda =
      sugodata_to_graph_error(B, Split, c, "Splitting Zeeman (Sonda Hall) ",
                              "Campo Magnetico ", "Split ");
  TGraphErrors *graph_exp = sugodata_to_graph_error(
      B_e, Split, c, "Splitting Zeeman ", "Campo Magnetico ", "Split ");

  TF1 *fit1 = new TF1("fit", "[0] * x + [1]", Split.min().value - 1,
                      Split.max().value + 1);
  fit1->SetLineColor(3);
  TF1 *fit2 = new TF1("fit", "[0] * x + [1]", Split.min().value - 1,
                      Split.max().value + 1);
  fit2->SetLineColor(2);

  graph_sonda->SetTitle("");
  graph_sonda->GetXaxis()->SetRangeUser(Split.min().value, Split.max().value);
  graph_exp->GetXaxis()->SetRangeUser(Split.min().value, Split.max().value);

  graph_sonda->Fit(fit1, "MRE+");
  graph_exp->Fit(fit2, "MRE+");
  multi->Add(graph_sonda);
  multi->Add(graph_exp);

  multi->Draw("AP");
  c->Update();

  gStyle->SetOptFit(1111);

  return 0;
}
