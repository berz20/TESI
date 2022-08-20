#include "../include/root.h"
#include <sstream>

TGraphErrors *sugodata_to_graph_error(Sugodata &x, Sugodata &y, TCanvas *c, std::string title, std::string x_title, std::string y_title){
    if(c == nullptr)
        c = new TCanvas();

    c->cd();
    c->SetGrid();
    
    TGraphErrors* graph = new TGraphErrors(x.size(), x.valuesSharedPtr().get(), y.valuesSharedPtr().get(), x.errorsSharedPtr().get(), y.errorsSharedPtr().get());
    graph->SetTitle(title.c_str());


    if(x.unit->compatible(units::adimensional)){
        graph->GetXaxis()->SetTitle(x_title.c_str());
    }
    else{
        std::stringstream unitx;
        x.unit->print(0, unitx);
        graph->GetXaxis()->SetTitle((x_title + "[" + unitx.str() + " ]").c_str());
    }

    if(y.unit->compatible(units::adimensional)){
        graph->GetYaxis()->SetTitle(y_title.c_str());
    }
    else{
        std::stringstream unity;
        y.unit->print(0, unity);
        graph->GetYaxis()->SetTitle((y_title + "[" + unity.str() + " ]").c_str());
    }

    return graph;
}

TH1F *sugodata_xy_to_th1f(Sugodata &x, Sugodata &y, TCanvas *c, double bin_width, std::string title, std::string x_title, std::string y_title){
    const double start = x.min().value * 0.9 ;
    const double end = x.max().value * 1.1 ;

    const double dispersione_max = x.max().value - x.min().value;
    double ampiezza_classe = bin_width ;
    const int n_classi = ceil(dispersione_max / ampiezza_classe);

    const double dispersione_ist = n_classi * ampiezza_classe;

    std::cout << "N Classi = " << n_classi << std::endl 
         << "Approssimato da: " << dispersione_max << " / " << ampiezza_classe << " = " << dispersione_max/ampiezza_classe << std::endl
         << "Dispersione istogramma = " << dispersione_ist << std::endl << std::endl;

    if(c == nullptr)
        c = new TCanvas();

    c->cd();
    
    TH1F *HistGr = new TH1F(title.c_str(), title.c_str(), n_classi, start, end); 

    for(size_t i=0; i<x.size(); i++){
        HistGr->SetBinContent(x[i].value, y[i].value);
    }

    std::stringstream unity, unitx;
    x.unit->print(0, unitx);
    y.unit->print(0, unity);

    HistGr->SetTitle(title.c_str());

    if(x.unit->compatible(units::adimensional)){
        HistGr->GetXaxis()->SetTitle(x_title.c_str());
    }
    else{
        std::stringstream unitx;
        x.unit->print(0, unitx);
        HistGr->GetXaxis()->SetTitle((x_title + "[" + unitx.str() + " ]").c_str());
    }

    if(y.unit->compatible(units::adimensional)){
        HistGr->GetYaxis()->SetTitle(y_title.c_str());
    }
    else{
        std::stringstream unity;
        y.unit->print(0, unity);
        HistGr->GetYaxis()->SetTitle((y_title + "[" + unity.str() + " ]").c_str());
    }

    return HistGr;
}

TH1F *sugodata_frequency_th1f(Sugodata &x, TCanvas *c, double bin_width, std::string title, std::string x_title, std::string y_title){
    const double start = x.min().value;
    const double end = x.max().value;

    const double dispersione_max = x.max().value - x.min().value;
    double ampiezza_classe = bin_width ;
    const int n_classi = ceil(dispersione_max / ampiezza_classe);

    const double dispersione_ist = n_classi * ampiezza_classe;

    std::cout << "N Classi = " << n_classi << std::endl 
         << "Approssimato da: " << dispersione_max << " / " << ampiezza_classe << " = " << dispersione_max/ampiezza_classe << std::endl
         << "Dispersione istogramma = " << dispersione_ist << std::endl << std::endl;

    if(c == nullptr)
        c = new TCanvas();

    c->cd();
    
    TH1F *HistGr = new TH1F(title.c_str(), title.c_str(), n_classi, start, end); 

    for(size_t i=0; i<x.size(); i++){
        HistGr->Fill(x[i].value, 1);
    }

    std::stringstream unitx;
    x.unit->print(0, unitx);

    HistGr->SetTitle(title.c_str());

    if(x.unit->compatible(units::adimensional)){
        HistGr->GetXaxis()->SetTitle(x_title.c_str());
    }
    else{
        std::stringstream unitx;
        x.unit->print(0, unitx);
        HistGr->GetXaxis()->SetTitle((x_title + "[" + unitx.str() + " ]").c_str());
    }

    return HistGr;
}
