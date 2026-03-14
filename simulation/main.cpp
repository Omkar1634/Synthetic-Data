#include "functions.h"
#include <random> 
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

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
#define nt 1.33

// ============================================================
// PROGRESS REPORTING  (thread-safe)
// ============================================================
// Tracks how many parameter combinations have finished.
// Workers increment this atomically; after every REPORT_INTERVAL
// completions the reporting thread prints a progress line.
// ============================================================
static std::atomic<long long> g_completed{0};   // total rows finished so far
static long long               g_total     = 0; // set in main() before threads start
static const long long         REPORT_INTERVAL = 5000; // print every N rows

// Called once per completed row inside ProcessAndWriteBioSkin.
// Only the thread whose increment lands on a multiple of REPORT_INTERVAL
// actually prints, keeping output clean with no extra mutex.
static void reportProgress(std::chrono::steady_clock::time_point programStart)
{
    long long done = g_completed.fetch_add(1, std::memory_order_relaxed) + 1;

    // Print on every REPORT_INTERVAL boundary, and also on the very last row
    if (done % REPORT_INTERVAL == 0 || done == g_total)
    {
        auto now     = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - programStart).count();

        double pct   = 100.0 * done / g_total;
        double eta   = (done < g_total)
                       ? elapsed * (g_total - done) / done
                       : 0.0;

        // Format ETA as mm:ss
        int eta_min  = static_cast<int>(eta) / 60;
        int eta_sec  = static_cast<int>(eta) % 60;

        std::cout << std::fixed << std::setprecision(1)
                  << "[Progress] "
                  << done << " / " << g_total
                  << "  (" << pct << "%)"
                  << "  elapsed: " << std::setprecision(1) << elapsed << "s"
                  << "  ETA: " << eta_min << "m " << eta_sec << "s"
                  << std::endl;
    }
}


double MonteCarlo(double sc_mua, double sc_mus, double sc_g, double sc_thickness,
                  double epi_mua, double epi_mus, double epi_g,
                  double derm_mua, double derm_mus, double derm_g,
                  double epidermis_thickness) {
    int Nphotons = 10000;
    double sc_albedo   = sc_mus  / (sc_mus  + sc_mua);
    double epi_albedo  = epi_mus / (epi_mus + epi_mua);
    double derm_albedo = derm_mus / (derm_mus + derm_mua);
    double epi_bottom  = sc_thickness + epidermis_thickness;  // z-depth where dermis begins
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
        double mua = sc_mua;    // photon starts at z=0, inside stratum corneum
        double mus = sc_mus;
        double albedo = sc_albedo;
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
                // --- FIX 2: Recover pre-step position, then step exactly to z=0 ---
                double x_old = x - s * ux;
                double y_old = y - s * uy;
                double z_old = z - s * uz;  // z_old > 0 (photon was inside tissue)

                // Correct path length from pre-step position to the surface (z=0)
                double s1 = z_old / (-uz);  // uz < 0, so -uz > 0; s1 > 0

                // Exact surface position
                double x_surf = x_old + s1 * ux;
                double y_surf = y_old + s1 * uy;
                // z at surface = 0

                // --- FIX 1: Fresnel at tissue->air interface (nt=1.33 -> 1.0) ---
                double internal_reflectance = RFresnel(nt, 1.0, -uz);
                double external_reflectance = 1.0 - internal_reflectance;

                r = sqrt(x_surf * x_surf + y_surf * y_surf);
                ir = static_cast<int>(r / dr);
                if (ir >= NR) {
                    ir = NR;
                }
                if (ir < 0) {
                    ir = 0;
                }
                ReflBin[ir] = ReflBin[ir] + (W * external_reflectance);
                W = internal_reflectance * W;

                // Reflect direction and continue remaining path from the surface
                uz = -uz;  // now positive (heading back into tissue)
                double s_remaining = s - s1;
                x = x_surf + s_remaining * ux;
                y = y_surf + s_remaining * uy;
                z = s_remaining * uz;  // z_surface = 0, so z = s_remaining * uz
            }

            if (z < sc_thickness) {
                mua = sc_mua;   mus = sc_mus;   albedo = sc_albedo;
            } else if (z < epi_bottom) {
                mua = epi_mua;  mus = epi_mus;  albedo = epi_albedo;
            } else {
                mua = derm_mua; mus = derm_mus; albedo = derm_albedo;
            }

            absorb = W * (1 - albedo);
            W = W - absorb;

            // Determine which g to use based on layer
            double current_g;
            if (z < sc_thickness) {
                current_g = sc_g;
            } else if (z < epi_bottom) {
                current_g = epi_g;
            } else {
                current_g = derm_g;
            }

            // Sample for costheta
            rnd = static_cast<double>(rand()) / RAND_MAX;
            if (current_g == 0.0) {
                costheta = 2.0 * rnd - 1.0;
            }
            else {
                double temp = (1.0 - current_g * current_g) / (1.0 - current_g + 2 * current_g * rnd);
                costheta = (1.0 + current_g * current_g - temp * temp) / (2.0 * current_g);
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
    for (int i = 0; i < NR; i++) {
    total_reflection += ReflBin[i];
    } 
    return total_reflection / Nphotons;
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
    std::vector<double> mua_epi_values;   // Store all wavelengths
    std::vector<double> mus_epi_values;
    std::vector<double> mua_derm_values;
    std::vector<double> mus_derm_values;
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

        double lambda_normalized = nm / 500.0;  // Normalize to 500nm
        double rayleigh_term = 0.48 * pow(lambda_normalized, -4.0);
        double mie_term = 0.52 * pow(lambda_normalized, -0.22);  // (1 - fRay) = 0.52
        double Us_epidermis = 36.4 * (rayleigh_term + mie_term);  // Result in cm⁻¹
        double Us_dermis = 0.65 * Us_epidermis;  // Dermis has ~65% scattering of epidermis (reduced from 0.75 to improve 600-700nm fit)
        double g = 0.62 + 0.00029 * (nm - 380.0);

        // ============================================================
        // STRATUM CORNEUM (SC) — 3rd layer (topmost), ~15 μm thick
        // Dead keratinocyte layer: high scattering, no chromophores.
        // Key contributor to 400–440 nm backscattering (~0.10–0.15 reflectance).
        // ============================================================
        const double sc_thickness = 0.0015;                          // cm (15 μm, fixed, not a free parameter)
        double sc_mua = alpha_base;                                   // background absorption only (no melanin, no Hb)
        double sc_mus = 100.0 * std::pow(400.0 / nm, 0.8);           // ~100 cm⁻¹ at 400 nm, falls with wavelength
        const double sc_g = 0.70;                                     // slightly more isotropic than epidermis (g=0.62)

        // ============================================================
        // MONTE CARLO LIGHT TRANSPORT  (3-layer: SC → epidermis → dermis)
        // ============================================================
        // T: epidermis thickness in cm (not including SC)

        double reflectance = MonteCarlo(sc_mua, sc_mus, sc_g, sc_thickness,
                                        Uepidermis, Us_epidermis, g,
                                        Udermis, Us_dermis, g,
                                        epidermis_thickness);

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

        // // Epidermis absorption (85)
        // for (double val : mua_epi_values) {
        //     row.push_back(val);
        // }
        
        // // Epidermis scattering (85)
        // for (double val : mus_epi_values) {
        //     row.push_back(val);
        // }
        
        // // Dermis absorption (85)
        // for (double val : mua_derm_values) {
        //     row.push_back(val);
        // }
        
        // // Dermis scattering (85)
        // for (double val : mus_derm_values) {
        //     row.push_back(val);
        // }
        // ===== ADD SPECTRAL REFLECTANCE VALUES (85 columns) =====
        for (double reflectance : reflectances) {
        row.push_back(reflectance);
        }
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


std::vector<double> generateSequence(double start, double end, int numSamples, double exponent) {
    std::vector<double> values;

    // Create uniform distribution from 0 to 1
    for (int i = 0; i < numSamples; ++i) {
        double uniformValue = static_cast<double>(i) / (numSamples - 1);  // Gives [0, 0.25, 0.5, 0.75, 1.0] for 5 samples
        
        // Apply the exponent (cubic = 3, quartic = 4)
        double exponentiatedValue = std::pow(uniformValue, exponent);  // For cubic: x³
        
        // Scale from [0,1] to [start,end]
        double scaledValue = start + (end - start) * exponentiatedValue;
        
        values.push_back(scaledValue);
    }

    return values;
}


std::mutex mtx; // For synchronizing output
std::mutex task_mtx; // Mutex for task queue
std::condition_variable cv; // Condition variable for the task queue

std::queue<std::function<void()>> tasks;

bool finished = false;



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

// ============================================================
// THREAD POOL
// ============================================================
std::mutex mtx;
std::mutex task_mtx;
std::condition_variable cv;
std::queue<std::function<void()>> tasks;
bool finished = false;

// programStart is passed in so reportProgress can compute elapsed time
static std::chrono::steady_clock::time_point g_programStart;

void ProcessAndWriteBioSkin(std::ofstream& outputFile,
                             double melanin_concentration, double blood_concentration,
                             double melanin_blend,         double blood_oxy,
                             double epidermis_thickness)
{
    std::vector<double> row = Bioskin(melanin_concentration, blood_concentration,
                                      melanin_blend, blood_oxy, epidermis_thickness);
    if (!row.empty()) {
        std::lock_guard<std::mutex> lock(mtx);
        WriteRowToCSV(outputFile, row);
    }

    // ---- NEW: report progress after every completed task ----
    reportProgress(g_programStart);
}

void worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(task_mtx);
            cv.wait(lock, [] { return !tasks.empty() || finished; });
            if (tasks.empty() && finished) return;
            task = std::move(tasks.front());
            tasks.pop();
        }
        task();
    }
}

// ============================================================
// MAIN
// ============================================================
int main() {
    // Parameter grid (unchanged)
    std::vector<double> CmValues      = generateSequence(0.013, 0.50, 20, 2);
    std::vector<double> ChValues      = generateSequence(0.01, 0.32, 20, 2);
    std::vector<double> BmValues      = generateSequence(0.0,  1.0,   5, 2);
    std::vector<double> BloodOxyValues= generateSequence(0.60, 0.98, 10, 1);
    std::vector<double> TValues       = generateSequence(0.005,0.020,  3, 1);

    g_total = (long long)CmValues.size() * ChValues.size() * BmValues.size()
            * BloodOxyValues.size() * TValues.size();

    std::cout << "Size of cartesian product: " << g_total << std::endl;
    std::cout << "Progress will be printed every " << REPORT_INTERVAL << " rows.\n" << std::endl;

    // Output file
    auto now   = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    std::string outputFilename = "lut_rgb_60k_" + ss.str() + ".csv";
    std::ofstream outputFile(outputFilename);

    // ---- Set global start time BEFORE launching threads ----
    g_programStart = std::chrono::steady_clock::now();
    auto wall_start = g_programStart;          // kept for final summary

    WriteHeaderToCSVBioWithSpectral(outputFile);

    // Thread pool
    const int numThreads = std::thread::hardware_concurrency();
    std::cout << "Using " << numThreads << " CPU threads.\n" << std::endl;
    std::vector<std::thread> workers;
    for (int i = 0; i < numThreads; i++) workers.emplace_back(worker);

    // Enqueue all tasks
    for (auto cm : CmValues)
      for (auto ch : ChValues)
        for (auto bm : BmValues)
          for (auto blood_oxy : BloodOxyValues)
            for (auto t : TValues) {
                auto task = [&outputFile, cm, ch, bm, blood_oxy, t]() {
                    ProcessAndWriteBioSkin(outputFile, cm, ch, bm, blood_oxy, t);
                };
                {
                    std::unique_lock<std::mutex> lock(task_mtx);
                    tasks.push(task);
                    cv.notify_one();
                }
            }

    // Signal done & join
    { std::unique_lock<std::mutex> lock(task_mtx); finished = true; cv.notify_all(); }
    for (auto& w : workers) w.join();

    outputFile.close();

    // Final summary
    double total_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start).count();
    std::cout << "\n=== Done ===" << std::endl;
    std::cout << "Total rows written : " << g_completed.load() << " / " << g_total << std::endl;
    std::cout << "Total elapsed time : " << std::fixed << std::setprecision(2)
              << total_elapsed << " seconds ("
              << static_cast<int>(total_elapsed)/60 << "m "
              << static_cast<int>(total_elapsed)%60 << "s)" << std::endl;
    std::cout << "Output file        : " << outputFilename << std::endl;

    return 0;
}

























// int main() {
//     double step_size = 5;
//     int numSamples = 15;
    
    
//     //Generate parameter ranges
    
//     std::vector<double> CmValues = generateSequence(0.0, 0.50, 20, 2);      // 1% to 50%
//     std::vector<double> ChValues = generateSequence(0.02, 0.32, 20, 2);      // 2% to 20% (raised min to ensure haemoglobin features are visible)
//     std::vector<double> BmValues = generateSequence(0.0, 1.0, 5, 2);         // 50% to 100%
//     std::vector<double> BloodOxyValues = generateSequence(0.60, 0.98, 10, 1); // 60% to 98%
//     std::vector<double> TValues = generateSequence(0.005, 0.020, 3, 1);      // 50μm to 200μm



//     std::cout << "Size of cartesian product: " << 
//         CmValues.size() * ChValues.size() * BmValues.size() * 
//         BloodOxyValues.size() * TValues.size() << std::endl;

//     // Get current time
//     auto now = std::chrono::system_clock::now();
//     std::time_t now_c = std::chrono::system_clock::to_time_t(now);

//     // Format datetime
//     std::stringstream ss;
//     ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    
//     std::string outputFilename = "lut_rgb_60k"+ ss.str() + ".csv";
//     std::ofstream outputFile(outputFilename);

//     // Start timers
//     auto start = std::chrono::high_resolution_clock::now();

//     WriteHeaderToCSVBioWithSpectral(outputFile);

//     // Thread pool setup
//     const int numThreads = std::thread::hardware_concurrency();
//     std::vector<std::thread> workers;

//     for (int i = 0; i < numThreads; i++) {
//         workers.push_back(std::thread(worker));
//     }

//     // Generate all combinations   
//     for (auto cm : CmValues) {
//         for (auto ch : ChValues) {
//             for (auto bm : BmValues) {
//                 for (auto blood_oxy : BloodOxyValues) {  // RENAMED from bh!
//                     for (auto t : TValues) {
//                         auto task = [&, cm, ch, bm, blood_oxy, t]() {
//                             ProcessAndWriteBioSkin(outputFile, cm, ch, bm, blood_oxy, t);
//                         };

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

//     // Signal completion
//     {
//         std::unique_lock<std::mutex> lock(task_mtx);
//         finished = true;
//         cv.notify_all();
//     }

//     // Wait for all workers to finish
//     for (auto& worker : workers) {
//         worker.join();
//     }

//     outputFile.close();
    
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

//     return 0;
// }






















