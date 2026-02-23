//function to calculate fresnel reflection coefficient
//math
#include "functions.h"




double getDeoxyHbValue(int wavelength)
{
    // Convert the vectors to maps for efficient access:
    std::map<int, double> deoxy_hb_map(deoxy_hb.begin(), deoxy_hb.end());
    if (deoxy_hb_map.find(wavelength) != deoxy_hb_map.end()) {
        return deoxy_hb_map[wavelength];

    }
    else {
        return 0.0;
    }
}

double getOxyHbValue(int wavelength)
{
    std::map<int, double> oxy_hb_map(oxy_hb.begin(), oxy_hb.end());

    if (oxy_hb_map.find(wavelength) != oxy_hb_map.end()) {
        return oxy_hb_map[wavelength];
    }
    else {
        return 0.0;
    }
}

double RFresnel(double n1, double n2, double cosT1) {
    double r = 0.0;
    double cosT2 = 0.0;
    const double COSZERO = 1.0 - 1.0e-12;
    const double COS90D = 1.0 * pow(10, -6);

    if (n1 == n2) { // Matched boundary
        r = 0.0;
        cosT2 = cosT1;
    }
    else if (cosT1 > COSZERO) { // Normal incident
        cosT2 = 0.0;
        r = (n2 - n1) / (n2 + n1);
        r *= r;
    }
    else if (cosT1 < COS90D) { // Very slant
        cosT2 = 0.0;
        r = 1.0;
    }
    else { // General case
        double sinT1 = std::sqrt(1 - cosT1 * cosT1);
        double sinT2 = n1 * sinT1 / n2;

        if (sinT2 >= 1.0) {
            r = 1.0;
            cosT2 = 0.0;
        }
        else {
            cosT2 = std::sqrt(1 - sinT2 * sinT2);
            double cosAP = cosT1 * cosT2 - sinT1 * sinT2;
            double cosAM = cosT1 * cosT2 + sinT1 * sinT2;
            double sinAP = sinT1 * cosT2 + cosT1 * sinT2;
            double sinAM = sinT1 * cosT2 - cosT1 * sinT2;
            r = 0.5 * sinAM * sinAM * (cosAM * cosAM + cosAP * cosAP) / (sinAP * sinAP * cosAM * cosAM);
        }
    }
    return r;
}

double xFit_1931(double wave)
{
    double t1 = (wave - 442.0) * ((wave < 442.0) ? 0.0624 : 0.0374);
    double t2 = (wave - 599.8) * ((wave < 599.8) ? 0.0264 : 0.0323);
    double t3 = (wave - 501.1) * ((wave < 501.1) ? 0.0490 : 0.0382);
    return 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2)
        - 0.065 * exp(-0.5 * t3 * t3);
}

double yFit_1931(double wave)
{
    double t1 = (wave - 568.8) * ((wave < 568.8) ? 0.0213 : 0.0247);
    double t2 = (wave - 530.9) * ((wave < 530.9) ? 0.0613 : 0.0322);
    return 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2);
}

double zFit_1931(double wave)
{
    double t1 = (wave - 437.0) * ((wave < 437.0) ? 0.0845 : 0.0278);
    double t2 = (wave - 459.0) * ((wave < 459.0) ? 0.0385 : 0.0725);
    return 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2);
}

double gamma_correction(double C) {
    double abs_C = std::abs(C);
    if (abs_C > 0.0031308) {
        return 1.055 * std::pow(abs_C, 1.0 / 2.4) - 0.055;
    }
    else {
        return 12.92 * C;
    }
}

std::vector<double> XYZ_to_sRGB(std::vector<double> xyz, int step_size) {
    double x = xyz[0]/step_size;
    double y = xyz[1]/step_size;
    double z = xyz[2]/step_size;

    // double x = xyz[0];
    // double y = xyz[1];
    // double z = xyz[2];

    std::vector<std::vector<double>> mat3x3 = {
        {3.2406, -1.5372, -0.4986},
        {-0.9689, 1.8758, 0.0415},
        {0.0557, -0.204, 1.057}
    };

    double r = x * mat3x3[0][0] + y * mat3x3[0][1] + z * mat3x3[0][2];
    double g = x * mat3x3[1][0] + y * mat3x3[1][1] + z * mat3x3[1][2];
    double b = x * mat3x3[2][0] + y * mat3x3[2][1] + z * mat3x3[2][2];

    double r_gamma = gamma_correction(r);
  
    g = gamma_correction(g) * 255.0;
    b = gamma_correction(b) * 255.0;

    // Round to 2 decimal places
    r = std::round(r * 1000.0) / 1000.0;
    g = std::round(g * 1000.0) / 1000.0;
    b = std::round(b * 1000.0) / 1000.0;

    // Vectorized version
    std::vector<double> sRGB = {r, g, b};

    return sRGB;
}

std::vector<double> Get_RGB(std::vector<double> wavelengths, std::vector<double> reflectances, int step_size) {
    std::vector<double> total = { 0.0, 0.0, 0.0 };
    int index = 0;
    std::vector<double> sRGB = { 0.0, 0.0, 0.0 };
    for (double nm : wavelengths) {

        double reflectance = reflectances[index];
        double x = xFit_1931(nm) * reflectance;
        double y = yFit_1931(nm) * reflectance;
        double z = zFit_1931(nm) * reflectance;

        //XYZ to sRGB

        total[0] += x ;
        total[1] += y ;
        total[2] += z ;

        index++;
    }

    //clip values 0 to 255
    sRGB = XYZ_to_sRGB(total,step_size);

    return sRGB;
}


double getD65Value(int wavelength) {
    // D65_ILLUMINANT uses double keys, so convert int to double
    double wl = static_cast<double>(wavelength);
    
    // Direct map lookup (no need to convert to another map)
    auto it = D65_ILLUMINANT.find(wl);
    
    if (it != D65_ILLUMINANT.end()) {
        return it->second;
    }
    else {
        // Wavelength not in table - this shouldn't happen for 380-780nm range
        std::cerr << "Warning: D65 value not found for wavelength " << wavelength << "nm" << std::endl;
        return 100.0;  // Return reasonable default instead of 0.0
    }
}

void WriteRowToCSV(std::ofstream& file, const std::vector<double>& row) {
    for (size_t i = 0; i < row.size(); ++i) {
        file << row[i];
        if (i < row.size() - 1) {
            file << ",";
        }
    }
    file << "\n";
}
void WriteHeaderToCSV(std::ofstream& file)
{
    //remove all white space and seperate by commas
    //Cm,Ch,Bm,Bh,T,sR,sG,sB
    file << "Cm,Ch,Bm,Bh,T,sR,sG,sB\n";
    
}


void WriteHeaderToCSVBio(std::ofstream& file)
{
    // Updated header with XYZ values before sRGB conversion
    file << "melanin_concentration(Cm),blood_concentration(Ch),melanin_blend(Bm),BloodOxy,epidermis_thickness(T),X,Y,Z,sR,sG,sB\n";
}
// void WriteHeaderToCSV(std::ofstream& file)
// {
//     //remove all white space and seperate by commas
//     //Cm,Ch,Bm,Bh,T,sR,sG,sB
//     // file << "Cm,Ch,Bm,Bh,T,sR,sG,sB\n";
//     file << "melanin_concentration(Cm),blood_concentration(Ch),melanin_blend(Bm),BloodOxy,epidermis_thickness(T),X,Y,Z,sR,sG,sB\n";
    
// }




// void WriteHeaderToCSV(std::ofstream& file)
// {
//     // Updated header with CTCAE metadata
//     file << "sample_id,fitzpatrick_type,toxicity_type,";
//     file << "Cm,Ch,Bm,Bh,T,";
//     file << "baseline_Cm,baseline_Ch,";
//     file << "ctcae_grade,sR,sG,sB\n";
// }

std::pair<double, double> calculate_absorption_coefficient(double wavelength) {
    // Check if wavelength matches any of the known values
    for (size_t i = 0; i < sizeof(wavelengths) / sizeof(wavelengths[0]); ++i) {
        if (wavelength == wavelengths[i]) {
            double e_HbO2 = deoxy_data[i];
            double e_Hb = oxy_data[i];
            return std::make_pair(e_HbO2, e_Hb);
        }
    }

    // If no match found, calculate using raw coefficients
    int i = std::lower_bound(wavelengths, wavelengths + sizeof(wavelengths) / sizeof(wavelengths[0]), wavelength) - wavelengths;
    if (i == 0) {
        i = 1.0;
    }
    else if (i == sizeof(wavelengths) / sizeof(wavelengths[0])) {
        i = sizeof(wavelengths) / sizeof(wavelengths[0]) - 1;
    }

    // Linear interpolation for e_HbO2
    double x0 = wavelengths[i - 1], x1 = wavelengths[i];
    double y0_HbO2 = deoxy_data[i - 1], y1_HbO2 = deoxy_data[i];
    double e_HbO2 = y0_HbO2 + (y1_HbO2 - y0_HbO2) * (wavelength - x0) / (x1 - x0);

    // Linear interpolation for e_Hb
    double y0_Hb = oxy_data[i - 1], y1_Hb = oxy_data[i];
    double e_Hb = y0_Hb + (y1_Hb - y0_Hb) * (wavelength - x0) / (x1 - x0);

    return std::make_pair(e_HbO2, e_Hb);
}

// ============================================================
// CTCAE GRADING FUNCTIONS - 
// ============================================================

int AssignErythemaGrade(double blood_volume) {
    /*
    Assign CTCAE grade based on blood volume fraction
    Based on research: Δa* color space changes correlate with blood
    
    Grade 0: Normal (< 2.5% blood)
    Grade 1: Faint erythema (2.5-6% blood)
    Grade 2: Moderate erythema (6-12% blood)
    Grade 3: Severe erythema (12-25% blood)
    Grade 4: Life-threatening (> 25% blood)
    */
    
    if (blood_volume < 0.025) {
        return 0;  // Normal/baseline
    }
    else if (blood_volume < 0.060) {
        return 1;  // Mild/faint erythema
    }
    else if (blood_volume < 0.120) {
        return 2;  // Moderate erythema
    }
    else if (blood_volume < 0.250) {
        return 3;  // Severe erythema
    }
    else {
        return 4;  // Life-threatening (rare)
    }
}

int AssignHyperpigmentationGrade(double melanin_volume, double baseline_melanin) {
    /*
    Assign CTCAE grade based on percent increase from baseline
    
    Grade 0: < 25% increase
    Grade 1: 25-50% increase (mild darkening)
    Grade 2: 50-80% increase (moderate darkening)
    Grade 3: > 80% increase (severe darkening)
    */
    
    // Calculate percent increase
    double percent_increase = ((melanin_volume - baseline_melanin) / baseline_melanin) * 100.0;
    
    if (percent_increase < 25.0) {
        return 0;
    }
    else if (percent_increase < 50.0) {
        return 1;
    }
    else if (percent_increase < 80.0) {
        return 2;
    }
    else {
        return 3;
    }
}

double GetHemoOxygenByGrade(int grade) {
    /*
    Get hemoglobin oxygenation (Bh parameter) based on inflammation severity
    
    Healthy tissue: 80% oxygenated
    Moderate inflammation: 75% oxygenated
    Severe inflammation: 70% oxygenated
    
    This affects color: high O2 = bright red, low O2 = dark red/purple
    */
    
    if (grade <= 1) {
        return 0.80;  // Grade 0-1: Healthy, well-oxygenated
    }
    else if (grade == 2) {
        return 0.75;  // Grade 2: Moderate inflammation
    }
    else {
        return 0.70;  // Grade 3-4: Severe inflammation, poor oxygenation
    }
}


