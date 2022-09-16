#include "sugodata/include/sugodata.h"
#include "sugodata/include/root.h"
#include "sugodata/Sugodatum.cpp"
#include "sugodata/Sugodata.cpp"
#include "sugodata/Unit.cpp"
#include "sugodata/UnitsTable.cpp"
#include "sugodata/utils.cpp"
#include "sugodata/root/root.cpp"
#include <filesystem>
#include <TStyle.h>
#include "TH1F.h"
#include "TF1.h"
#include "TLine.h"
#include "TMath.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TPaveStats.h"
#include "TMultiGraph.h"
#include "TLegend.h"
/* Dir must end with / */
void printCanvas(TCanvas *c, std::string canvas_name, std::string dir){
    /* if(!(std::filesystem::exists(dir) || std::filesystem::is_directory(dir))) */
        /* std::filesystem::create_directories(dir); */

    c->Print((dir + canvas_name + ".pdf").c_str());
}

Sugodatum SPESSORE_ETALON(3, 0, multiple(units::meter, -3));

Sugodatum INDICE_RIFRAZIONE_NORMALE(1.4560, 0, units::adimensional);
Sugodatum INDICE_RIFRAZIONE_ANOMALO(1.4519, 0, units::adimensional);

Sugodatum LAMBDA_TRANSIZIONE_NORMALE(643.847, 0, multiple(units::meter, -9));
Sugodatum LAMBDA_TRANSIZIONE_ANOMALO(508.588, 0, multiple(units::meter, -9));

Sugodatum COSTANTE_PLANCK(6.626070154e-34, 0, units::joule * units::second);
Sugodatum VELOCITA_LUCE(299792458, 0, units::meter / units::second);

Sugodatum MAGNETONE_BOHR_TEORICO(9.27400999457e-24, 0, units::joule / units::tesla);
Sugodatum GYROMAGNETIC_RATIO_DIAMOND(176086.2682, 0, multiple(units::hertz, 9) / multiple(units::tesla, -3) );
Sugodatum PI(3.1415926535897932384626433, 0, units::adimensional);
