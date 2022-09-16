#ifndef sugodata_utils_templates_h
#define sugodata_utils_templates_h

#include "Sugodata.h"
#include <string>

template <class ... Args>
bool sugodata_parse_file_data(std::string filename, std::string separator, Args&... args){
    std::vector<std::reference_wrapper<Sugodata>> sugodata = {args...};
    
    std::ifstream ifile(filename);
    if(!ifile.good())
        return false;

    std::string line;
    while(std::getline(ifile, line)){
        size_t sep, last_sep = 0;
        for(size_t i=0; i<sugodata.size(); i++){
            sep = line.find(separator, last_sep);
            if(sep == std::string::npos){
                sugodata.at(i).get().add(Sugodatum(std::stod(line.substr(last_sep)), 0));
            }
            else
                sugodata.at(i).get().add(Sugodatum(std::stod(line.substr(last_sep, sep-last_sep)), 0));
            last_sep = sep+1;
        }
    }

    ifile.close();
    return true;
}

template <class ... Args>
bool sugodata_parse_file_data_errors(std::string filename, std::string separator, Args &... args){
    std::vector<std::reference_wrapper<Sugodata>> sugodata = {args...};
    
    std::ifstream ifile(filename);
    if(!ifile.good())
        return false;

    std::string line;
    while(std::getline(ifile, line)){
        size_t sep, sep2, last_sep = 0;
        for(size_t i=0; i<sugodata.size(); i++){
            sep = line.find(separator, last_sep);
            sep2 = line.find(separator, sep+1);
            if(sep2 == std::string::npos){
                sugodata.at(i).get().add(Sugodatum(std::stod(line.substr(last_sep, sep-last_sep)), std::stod(line.substr(sep+1))));
            }
            else
                sugodata.at(i).get().add(Sugodatum(std::stod(line.substr(last_sep, sep-last_sep)), std::stod(line.substr(sep+1, sep2-sep-1))));
            last_sep = sep2+1;
        }
    }

    ifile.close();
    return true;
}

template <class ... Args>
void print_as_latex_table(std::ostream &os, Args &... args){
    std::vector<std::reference_wrapper<Sugodata>> sugodata = {args...};
    
    const int n_columns = sizeof...(args);
    
    os << "\\begin{tabular}{";
    for(int i=0; i<n_columns; i++)
        os << "|c";
    
    os << "|} \\hline\n";
    
	/* Elements say if errors or ums are constants in order to put them in the headers instead of in the lines */
    std::vector<bool> constants;	   

	for(int i=0; i<n_columns; i++)
		constants.push_back(constant_errors(sugodata.at(i).get()));

	/* Print Headers */
    for(int i=0; i<n_columns; i++){
        if(constants.at(i) == false){
			os << " $ G $ \\ $ [" << *sugodata.at(i).get().unit << " ] $";
		}
        else{
            os << "$ G $ \\ $ \\pm $ \\ $ ";
            sugodata.at(i).get().get(0).printToSignificantDigits(-2, -1, "", os);
            os << " $ \\ $ [" << sugodata.at(i).get().unit << "] $"; 
		}
    	if(i != n_columns - 1) os << " & ";
    }
    os << " \\\\ \\hline\n"; 
   
    for(size_t j=0; j<sugodata.at(0).get().size(); j++){
        for(int i=0; i<n_columns; i++){
            sugodata.at(i).get().get(j).printToSignificantDigits(0, -1, " \\ $ \\pm \\ $", os);
            if(i != n_columns - 1) os << " & ";
        }   
            
        os << " \\\\ \\hline\n";
    }

    os << "\\end{tabular}\n";
}

#endif // !sugodata_utils_templates_h
