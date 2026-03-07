#include "functions.h"
#include <random> 

std::random_device rd; 
std::mt19937 gen(rd()); 
std::uniform_real_distribution<> dis(0.0, 1.0);

#define Nbins 1000
#define Nbinsp1 1001
#define PI 3.1415926
#define LIGHTSPEED 2.997925e10
#define ALIVE 1
#define DEAD 0
#define THRESHOLD 0.01
#define CHANCE 0.1
#define COS90D 1e-6
#define ONE_MINUS_COSZERO 1e-12
#define COSZERO (1.0 - 1.0e-12)
#define g 0.9
#define nt 1.33


double MonteCarlo(double epi_mua, double epi_mus, double derm_mua, double derm_mus, double epidermis_thickness) {
    int Nphotons = 1000;
    //double ReflBin[Nbinsp1];

    double epi_albedo = epi_mus / (epi_mus + epi_mua);
    double derm_albedo = derm_mus / (derm_mus + derm_mua);
    int NR = Nbins; //number of radial bins
    double radial_size = 2.5;
    double r = 0.0;
    // int ir = 0;
    double dr = radial_size / NR;
    //random seed
    std::vector<double> ReflBin(NR + 1, 0.0);
    srand(time(NULL));
    for (int i = 0; i < Nbinsp1; i++) {
        ReflBin[i] = 0;
    }

    for (int i_photon = 0; i_photon < Nphotons; i_photon++) {
        double W = 1.0;
        int photon_status = ALIVE;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        double ux, uy, uz;
        double costheta, sintheta, cospsi, sinpsi, psi, uxx, uyy, uzz;
        double s, rnd;
        int it, ir;
        double mua = epi_mua;
        double mus = epi_mus;
        double albedo = epi_albedo;
        double absorb;

        // Randomly set photon trajectory to yield an isotropic source.
        costheta = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
        sintheta = sqrt(1.0 - costheta * costheta);
        psi = 2.0 * PI * static_cast<double>(rand()) / RAND_MAX;
        // std::cout << "psi: " << psi << std::endl;
        ux = sintheta * cos(psi);
        uy = sintheta * sin(psi);
        uz = (fabs(costheta)); // fabs is 

        // Propagate one photon until it dies as determined by ROULETTE or reaches the surface
        it = 0;
        const int max_iterations = 10000;
        while (true) {
            it++;
            rnd = static_cast<double>(rand()) / RAND_MAX;
            // std::cout << "rnd: " << rnd << std::endl;
            while (rnd <= 0.0) {
                rnd = static_cast<double>(rand()) / RAND_MAX;

            }
            s = -log(rnd) / (mua + mus);
            x = x + (s * ux);
            y = y + (s * uy);
            z = z + (s * uz);

            if (uz < 0) {
                double s1 = fabs(z / uz);
                x = x - (s * ux);
                y = y - (s * uy);
                z = z - (s * uz);
                x = x + (s1 * ux);
                y = y + (s1 * uy);
                z = z + (s1 * uz);

                double internal_reflectance = RFresnel(1.0, nt, -uz);
                // std::cout << "internal_reflectance: " << internal_reflectance << std::endl;
                double external_reflectance = 1 - internal_reflectance;
                r = sqrt(x * x + y * y);
                ir = static_cast<int>(r / dr);
                if (ir >= NR) {
                    ir = NR;
                }
                if (ir < 0) {
                    ir = 0;
                }
                ReflBin[ir] = ReflBin[ir] + (W * external_reflectance);
                W = internal_reflectance * W;
                uz = -uz;
                x = (s - s1) * ux;
                y = (s - s1) * uy;
                z = (s - s1) * uz;
            }

            if (z < epidermis_thickness) {
                mua = epi_mua;
                mus = epi_mus;
                albedo = epi_albedo;
            }
            else {
                mua = derm_mua;
                mus = derm_mus;
                albedo = derm_albedo;
            }

            absorb = W * (1 - albedo);
            W = W - absorb;

            // Sample for costheta
            rnd = static_cast<double>(rand()) / RAND_MAX;
            if (g == 0.0) {
                costheta = 2.0 * rnd - 1.0;
            }
            else {
                double temp = (1.0 - g * g) / (1.0 - g + 2 * g * rnd);
                costheta = (1.0 + g * g - temp * temp) / (2.0 * g);
            }
            sintheta = sqrt(1.0 - costheta * costheta);

            // Sample psi
            psi = 2.0 * PI * static_cast<double>(rand()) / RAND_MAX;
            cospsi = cos(psi);
            if (psi < PI) {
                sinpsi = sqrt(1.0 - cospsi * cospsi);
            }
            else {
                sinpsi = -sqrt(1.0 - cospsi * cospsi);
            }

            if (1 - abs(uz) <= ONE_MINUS_COSZERO) {
                uxx = sintheta * cospsi;
                uyy = sintheta * sinpsi;
                uzz = costheta * copysign(uz, -1.0);
            }
            else {
                double temp = sqrt(1.0 - uz * uz);
                uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta;
                uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta;
                uzz = -sintheta * cospsi * temp + uz * costheta;
            }
            //update trajectory
            ux = uxx;
            uy = uyy;
            uz = uzz;
            if (W < THRESHOLD) {
                if (static_cast<double>(rand()) / RAND_MAX <= CHANCE) {
                    W = W / CHANCE;
                }
                else {
                    photon_status = DEAD;
                }
            }
            if (photon_status == DEAD || it > max_iterations) {
                break;
            }
        }
        if (i_photon >= Nphotons) {
            break;
        }
    }
    double total_reflection = 0.0;
    double max = 0.0;
    for (int i = 0; i < Nbinsp1; i++) {
        if (ReflBin[i] > max)
        {
            max = ReflBin[i]/Nphotons;
        }
        // std::cout << "ReflBin[" << each << "] = " << ReflBin[each] << std::endl;
        total_reflection += ReflBin[i] / Nphotons;

    }
    if (total_reflection > 1.0)
    {
	    	std :: cout << "total_reflection: " << total_reflection << std::endl;
            std::cout << "max: " << max << std::endl;
	}

    return total_reflection;
}
std::vector<float> generateDistribution(float minVal, float maxVal, int numSamples, double exponent = 1.0) {
    std::vector<float> values;
    float val;
    for (int i = 0; i < numSamples; ++i) {
        val = std::pow(minVal + (maxVal - minVal) * i / (numSamples - 1), exponent);
        values.push_back(round(val * 10000) / 10000); // Round to 4 decimal places
    }
    return values;
}
std::vector<double> generateArray(double a, double b, double s, bool print_results = false) {
    std::vector<double> result;
    for (double val = a; val <= b; val += s) {
        result.push_back(val);
    }
    if (print_results == true) {
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i] << std::endl;
        }
    }
    return result;
}
    

// std::vector<double> CalculateReflectanceRow(double Cm, double Ch, double Bm, double Bh, double T) {
//     // 380 to 780
//     int step_size = 10;
//     std::vector<double> wavelengths = generateArray(380, 780, step_size, false);

//     std::vector<double> reflectances(wavelengths.size());

//     //total
//     std::vector<double> total = { 0.0, 0.0, 0.0 };
//     int index = 0;
//     for (int nm : wavelengths) {
//         double alpha_base = 0.0244 + 8.53 * std::exp(-(nm - 154) / 66.2); // baseline absorption coefficient (Epidermis)
//         double alpha_em = 6.6 * std::pow(10, 10) * std::pow(nm, -3.33);  // eumelanin  absorption coefficient (Epidermis)
//         double alpha_pm = 2.9 * std::pow(10, 14) * std::pow(nm, -4.75); // pheomelanin  absorption coefficient (Epidermis)
//         // Calculate alpha_oxy and alpha_deoxy using appropriate data source
//         // double alpha_oxy = oxy_hb[index].second;
//         // double alpha_deoxy = deoxy_hb[index].second;
//         auto coefficients = calculate_absorption_coefficient(nm);
//         double gammaOxy = coefficients.first;
//         double gammaDeoxy = coefficients.second;
//         //double A = Bh*gammaOxy + (1 - Bh)*gammaDeoxy;

//         int wl = static_cast<int>(nm);
//         //using 
//         double epidermis = Cm * (Bm * alpha_em + (1 - Bm) * alpha_pm) + (1 - Cm) * alpha_base;
//         //alpha_derm_total = Ch*(Bh*alpha_oxy + (1-Bh)*alpha_deoxy) + (1-Ch)*self.get_alpha_base(wl)
//         double dermis = Ch *(Bh*gammaOxy + (1 - Bh)*gammaDeoxy) + (1 - Ch)*alpha_base;
//         //double dermis2 = Ch * (Bh * gammaOxy + (1 - Bh) * gemmaDeoxy) + (1 - Ch) * alpha_base;
//         //double dermis = Ch * (A + (1 - Ch) * alpha_base);
//         double scattering_epidermis = 14.74 * std::pow(nm, -0.22) + 2.22 * std::pow(10, 11) * std::pow(nm, -4.0);
//         double scattering_dermis = 0.75 * scattering_epidermis;
//         // Call MonteCarlo function here and store reflectance values
//         double reflectance = MonteCarlo(epidermis, scattering_epidermis, dermis, scattering_dermis, T);
//         wavelengths[index] = nm;
//         reflectances[index] = reflectance;
//         double x =  xFit_1931(nm);
//         double y =  yFit_1931(nm);
//         double z =  zFit_1931(nm);
//         double XYZ [3] = {x, y, z};

//         total[0] += x*reflectance;
//         total[1] += y*reflectance;
//         total[2] += z*reflectance;
//         index++;
//         //std::cout << "Total: " << total[0] << ", " << total[1] << ", " << total[2] << std::endl;
//     }
//     std::vector<double> sRGB0 = XYZ_to_sRGB(total, step_size);
//     std::vector<double> row;
//     if (!(sRGB0[0] > 255 || sRGB0[1] > 255 || sRGB0[2] > 255)) {
//         row.push_back(Cm);
//         row.push_back(Ch);
//         row.push_back(Bm);
//         row.push_back(Bh);
//         row.push_back(T);
//         row.push_back(sRGB0[0]);
//         row.push_back(sRGB0[1]);
//         row.push_back(sRGB0[2]);
//         total.clear();
//         sRGB0.clear();
//         return row;
//     }
//     else {
//         row.clear();
//         return row;
//     }
// }

std::vector<double> Bioskin(double melanin_concentration,  // Cm: Volume fraction of melanin in epidermis
    double blood_concentration,    // Ch: Volume fraction of blood in dermis
    double melanin_blend,          // Bm: Ratio of eumelanin (1.0) to pheomelanin (0.0)
    double blood_oxy,              // NEW NAME! Blood oxygenation level (0-1)
    double epidermis_thickness     // T: Thickness of epidermis in cm
)  {
    // Wavelength range: 380 to 800 nm with 5nm step
    int step_size = 5;
    std::vector<double> wavelengths = generateArray(380, 800, step_size, false);
    std::vector<double> reflectances(wavelengths.size());
    
    // Accumulator for XYZ color matching
    std::vector<double> total = {0.0, 0.0, 0.0};
    
    int index = 0;
    for (int nm : wavelengths) {
        // ============================================================
        // ABSORPTION COEFFICIENTS (μ_a) - Based on BioSkin Paper
        // ============================================================
        
        // Baseline absorption (tissue water, proteins, lipids)
        double alpha_base = 0.0244 + 8.53 * std::exp(-(nm - 154.0) / 66.2);
        
        // Melanin absorption spectra
        double alpha_eumelanin = 6.6e10 * std::pow(nm, -3.33);
        double alpha_pheomelanin = 2.9e14 * std::pow(nm, -4.75);
        
        // Beta-carotene absorption (minor chromophore)
        double alpha_carotene_epi = 2.1e-4;   // Epidermis
        double alpha_carotene_derm = 7.0e-5;  // Dermis
        
        // Hemoglobin absorption (from your data tables)
        auto hb_coefficients = calculate_absorption_coefficient(nm);
        double alpha_HbO2 = hb_coefficients.first;   // Oxygenated
        double alpha_Hb = hb_coefficients.second;     // Deoxygenated
        
        // ============================================================
        // EPIDERMIS ABSORPTION
        // ============================================================
        // Cm: melanin volume fraction (0-1)
        // Bm: melanin blend (0=pheomelanin, 1=eumelanin)
        // Note: Small amount of blood can perfuse into papillary dermis/epidermis junction
        
        double melanin_absorption = melanin_blend * alpha_eumelanin + (1.0 - melanin_blend) * alpha_pheomelanin;
        double Uepidermis = melanin_concentration * melanin_absorption + (1.0 - melanin_concentration) * (alpha_base + alpha_carotene_epi);
        
        // ============================================================
        // DERMIS ABSORPTION
        // ============================================================
        // Ch: blood volume fraction (0-1)
        // Bh: blood oxygenation (0=deoxygenated, 1=oxygenated)
        
        double blood_absorption = blood_oxy * alpha_HbO2 + (1.0 - blood_oxy) * alpha_Hb;
        double Udermis = blood_concentration * (blood_absorption + alpha_carotene_derm) + (1.0 - blood_concentration) * alpha_base;
        
        // ============================================================
        // SCATTERING COEFFICIENTS (μ_s) - Based on BioSkin Paper
        // ============================================================
        // Rayleigh scattering (wavelength^-4) + Mie scattering (wavelength^-0.22)
        
        double Us_epidermis = 2.22e11 * std::pow(nm, -4.0) + 14.74 * std::pow(nm, -0.22);
        double Us_dermis = 0.75 * Us_epidermis;  // Dermis has ~75% scattering of epidermis
        
        // ============================================================
        // MONTE CARLO LIGHT TRANSPORT
        // ============================================================
        // T: epidermis thickness in cm
        
        double reflectance = MonteCarlo(Uepidermis, Us_epidermis, Udermis, Us_dermis, epidermis_thickness);

        
        // Store result
        reflectances[index] = reflectance;

        double d65 = getD65Value(nm);
        
        // ============================================================
        // ACCUMULATE XYZ COLOR MATCHING FUNCTIONS
        // ============================================================
        double x = xFit_1931(nm);
        double y = yFit_1931(nm);
        double z = zFit_1931(nm);
        
        // ACCUMULATE WITH D65 AND STEP_SIZE (CRITICAL!)
        total[0] += x * reflectance * d65 * step_size;
        total[1] += y * reflectance * d65 * step_size;
        total[2] += z * reflectance * d65 * step_size;
        index++;
    }
    
    // ============================================================
    // STEP 2: NORMALIZE XYZ (NEW!)
    // ============================================================
    double normalization = 0.0;
    for (int nm : wavelengths) {
        double y = yFit_1931(nm);
        double d65 = getD65Value(nm);
        normalization += y * d65 * step_size;
    }
    
    // Scale to Y=100 for perfect white
    total[0] = 100.0 * total[0] / normalization;
    total[1] = 100.0 * total[1] / normalization;
    total[2] = 100.0 * total[2] / normalization;
    
    // ============================================================
    // CONVERT XYZ TO sRGB
    // ============================================================
    std::vector<double> sRGB = XYZ_to_sRGB(total, step_size);
    
    // ============================================================
    // BUILD OUTPUT ROW
    // ============================================================
    std::vector<double> row;
    
    // Only return valid colors (RGB values in range 0-255)
    if (!(sRGB[0] > 255 || sRGB[1] > 255 || sRGB[2] > 255)) {
        
        row.push_back(melanin_concentration);   // Melanin concentration
        row.push_back(blood_concentration);   // Blood concentration
        row.push_back(melanin_blend);   // Melanin blend
        row.push_back(blood_oxy);   // Blood oxygenation
        row.push_back(epidermis_thickness);    // Epidermis thickness
        // XYZ values (device-independent color space)
        row.push_back(total[0]);
        row.push_back(total[1]);
        row.push_back(total[2]);
        row.push_back(sRGB[0]); // Red
        row.push_back(sRGB[1]); // Green
        row.push_back(sRGB[2]); // Blue
    }
    
    return row;
}

std::vector<double> generateSequence(double start, double end, int numSamples, double root) {
    std::vector<double> values;

    double startRootValue = std::pow(start, 1.0 / root);
    double endRootValue = std::pow(end, 1.0 / root);
    double delta = (endRootValue - startRootValue) / (numSamples - 1);

    for (int i = 0; i < numSamples; ++i) {
        double valRoot = startRootValue + delta * i;
        values.push_back(std::pow(valRoot, root));
    }

    return values;
}


std::mutex mtx; // For synchronizing output
std::mutex task_mtx; // Mutex for task queue
std::condition_variable cv; // Condition variable for the task queue

std::queue<std::function<void()>> tasks;

bool finished = false;

// void ProcessAndWrite(std::ofstream& outputFile, double cm, double ch, double bm, double bh, double t) {
//     std::vector<double> row = CalculateReflectanceRow(cm, ch, bm, bh, t);
//     if (row.empty()) {
// 		return;
// 	}
//     mtx.lock();
//     WriteRowToCSV(outputFile, row);
//     //std::cout << "cm: " << cm << ", ch: " << ch << ", bm: " << bm << ", bh: " << bh << ", t: " << t << " \n" << std::endl;
//     mtx.unlock();
//     row.clear();

// }

void ProcessAndWriteBioSkin(std::ofstream& outputFile,double melanin_concentration,double blood_concentration,double melanin_blend,double blood_oxy,double epidermis_thickness) {
    std::vector<double> row = Bioskin( melanin_concentration, blood_concentration, melanin_blend, blood_oxy, epidermis_thickness );   
    if (row.empty()) {
        return;
    }
    mtx.lock();
    WriteRowToCSV(outputFile, row);
    //std::cout << "cm: " << cm << ", ch: " << ch << ", bm: " << bm << ", bh: " << bh << ", t: " << t << " \n" << std::endl;
    mtx.unlock();
    row.clear();

}
void worker() {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(task_mtx);

            cv.wait(lock, [] {
                return !tasks.empty() || finished;
                });

            if (tasks.empty() && finished) return;

            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

int main() {
    double step_size = 5;
    int numSamples = 15;
    
    
    //Generate parameter ranges
    // std::vector<double> CmValues = generateSequence(0.005, 0.5, 51, 2);  // melanin concentration Cm
    // std::vector<double> ChValues = generateSequence(0.001, 0.32, 51, 2);  // blood concentration Ch
    // std::vector<double> BmValues = generateSequence(0.001, 1.0, 5, 2);    // melanin blend bm
    // std::vector<double> BloodOxyValues = generateSequence(0.6, 0.98, 11, 1); // blood oxygenation Bh
    // std::vector<double> TValues = generateSequence(0.005, 0.025, 4, 1);   // epidermis thickness in cm

  // FOR DEBUGGING - Fast generation to verify the pipeline works
    std::vector<double> CmValues = generateSequence(0.01, 0.50, 15, 2);      // 1% to 50%
    std::vector<double> ChValues = generateSequence(0.002, 0.20, 10, 2);     // 0.2% to 20%
    std::vector<double> BmValues = generateSequence(0.5, 1.0, 3, 2);         // 50% to 100%
    std::vector<double> BloodOxyValues = generateSequence(0.60, 0.98, 5, 1); // 60% to 98%
    std::vector<double> TValues = generateSequence(0.005, 0.020, 3, 1);      // 50μm to 200μm

// Total: 15 × 10 × 3 × 5 × 3 = 6,750 combinations

   
    std::cout << "Size of cartesian product: " << 
        CmValues.size() * ChValues.size() * BmValues.size() * 
        BloodOxyValues.size() * TValues.size() << std::endl;
    
    std::string outputFilename = "lut_rgb_trial_values.csv";
    std::ofstream outputFile(outputFilename);

    // Start timers
    auto start = std::chrono::high_resolution_clock::now();

    WriteHeaderToCSVBio(outputFile);

    // Thread pool setup
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;

    for (int i = 0; i < numThreads; i++) {
        workers.push_back(std::thread(worker));
    }

    // Generate all combinations   
    for (auto cm : CmValues) {
        for (auto ch : ChValues) {
            for (auto bm : BmValues) {
                for (auto blood_oxy : BloodOxyValues) {  // RENAMED from bh!
                    for (auto t : TValues) {
                        auto task = [&, cm, ch, bm, blood_oxy, t]() {
                            ProcessAndWriteBioSkin(outputFile, cm, ch, bm, blood_oxy, t);
                        };

                        {
                            std::unique_lock<std::mutex> lock(task_mtx);
                            tasks.push(task);
                            cv.notify_one();
                        }
                    }
                }
            }
        }
    }

    // Signal completion
    {
        std::unique_lock<std::mutex> lock(task_mtx);
        finished = true;
        cv.notify_all();
    }

    // Wait for all workers to finish
    for (auto& worker : workers) {
        worker.join();
    }

    outputFile.close();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}





























// int main() {
//     double step_size = 5;
//     int numSamples = 15;
//     std::vector<double> CmValues = generateSequence(0.001, 1.0, numSamples, 3); //  melanin concentration
//     std::vector<double> ChValues = generateSequence(0.001, 1.0, numSamples, 4); //  hemoglobin concentration
//     std::vector<double> BmValues = generateSequence(0.0, 1.0, numSamples, 2); // melanin blend
//     std::vector<double> BhValues = generateSequence(0.0, 1.0, numSamples, 1);// hemoglobin blend
//     std::vector<double> TValues = generateSequence(0.01, 0.25, numSamples, 1); // thickness of skin in cm
//     std::cout << "size of cartesian product: " << CmValues.size() * ChValues.size() * BmValues.size() * BhValues.size() * TValues.size()<< std::endl;
//     std::string outputFilename = "lut_rgb.csv";
//     std::ofstream outputFile(outputFilename);

//     //start timer
//     auto start = std::chrono::high_resolution_clock::now();
//     int count = 0;

//     WriteHeaderToCSV(outputFile);

//     const int numThreads = std::thread::hardware_concurrency();
//     std::vector<std::thread> workers;

//     for (int i = 0; i < numThreads; i++) {
//         workers.push_back(std::thread(worker));
//     }

//     for (auto cm : CmValues) {
//         for (auto ch : ChValues) {
//             for (auto bm : BmValues) {
//                 for (auto bh : BhValues) {
//                     for (auto t : TValues) {
//                             auto task = [&, cm, ch, bm, bh, t]() {
//                                 ProcessAndWrite(outputFile, cm, ch, bm, bh, t);
//                             };

//                         {
//                             std::unique_lock<std::mutex> lock(task_mtx);
//                             tasks.push(task);
//                             cv.notify_one();
//                         }
//                     }
//                 }
//             }
//         }
//     }

//     {
//         std::unique_lock<std::mutex> lock(task_mtx);
//         finished = true;
//         cv.notify_all();
//     }

//     for (auto& worker : workers) {
//         worker.join();
//     }

//     outputFile.close();
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "elapsed time: " << elapsed.count() << " seconds" << std::endl;

//     outputFile.close();
//     return 0;

// }
// Generate parameter ranges
    // std::vector<double> CmValues = generateSequence(0.001, 1.0, 51, 3);  // melanin concentration
    // std::vector<double> ChValues = generateSequence(0.001, 1.0, 51, 4);  // blood concentration
    // std::vector<double> BmValues = generateSequence(0.0, 1.0, 5, 2);    // melanin blend
    // std::vector<double> BloodOxyValues = generateSequence(0.6, 0.95, 11, 1); // blood oxygenation (RENAMED & realistic range!)
    // std::vector<double> TValues = generateSequence(0.01, 0.25, 4, 1);   // epidermis thickness in cm
    

