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
#include <atomic>

// === Writing system ===
std::queue<std::vector<double>> writeQueue;
std::mutex write_mtx;
std::condition_variable write_cv;

// === Progress tracking ===
std::atomic<int> completedTasks(0);
int totalTasks = 0;

// === Control flags ===
bool writingFinished = false;

double MonteCarlo(
    double epi_mua, double epi_mus, double epi_g,
    double derm_mua, double derm_mus, double derm_g,
    double epidermis_thickness)
{
    const int    totalPhotons = 100000; 
    const double n_tissue     = 1.4;
    const double n_air        = 1.0;

    thread_local std::mt19937 gen(std::random_device{}());
    thread_local std::uniform_real_distribution<> dis(0.0, 1.0);

    const double epi_albedo  = epi_mus  / (epi_mus  + epi_mua);
    const double derm_albedo = derm_mus / (derm_mus + derm_mua);

    double total_reflection = 0.0;

    for (int i_photon = 0; i_photon < totalPhotons; i_photon++) {

        // ── Lambertian launch (paper section 3.2) ──────────────────────
        // Photon starts just inside the surface, direction cosine-weighted.
        // 2D walk: only track (r, z). Azimuthal symmetry assumed.
        // r is the radial distance from launch point.
        double uz = sqrt(dis(gen));             // cos-weighted, uz > 0 (into tissue)
        double ur = sqrt(1.0 - uz * uz);        // radial component (2D)

        double r = 0.0;
        double z = 0.0;
        double W = 1.0;

        double mua    = epi_mua;
        double mus    = epi_mus;
        double g      = epi_g;
        double albedo = epi_albedo;

        const int max_iter = 100000;

        for (int it = 0; it < max_iter; it++) {

            // ── Sample step length ──────────────────────────────────────
            double rnd = dis(gen);
            while (rnd <= 0.0) rnd = dis(gen);
            double mu_t = mua + mus;
            double s    = -log(rnd) / mu_t;

            // ── Find distance to nearest boundary ───────────────────────
            double s_bound  = 1e30;
            bool   hit_surf = false;   // true = top surface, false = epi/derm

            if (uz > 1e-10) {
                // Moving deeper: check epi→derm boundary
                if (z < epidermis_thickness) {
                    double s_epi = (epidermis_thickness - z) / uz;
                    if (s_epi < s_bound) { s_bound = s_epi; hit_surf = false; }
                }
            } else if (uz < -1e-10) {
                // Moving up: check top surface z = 0
                double s_top = -z / uz;
                if (s_top < s_bound) { s_bound = s_top; hit_surf = true; }
            }

            if (s_bound < s) {
                // ── Step to boundary ────────────────────────────────────
                r += s_bound * ur;
                z += s_bound * uz;
                double s_left = s - s_bound;

                if (hit_surf) {
                    // ── Top surface: Fresnel (tissue→air) ───────────────
                    double cos_i = fabs(uz);
                    double R     = RFresnel(n_tissue, n_air, cos_i);

                    if (dis(gen) > R) {
                        // Transmitted — photon escapes, record reflectance
                        total_reflection += W;
                        goto next_photon;
                    } else {
                        // Internal reflection — flip direction
                        uz = -uz;
                    }
                }
                // epi/derm boundary: no Fresnel (same n per paper)
                // just update optical properties at new z
                if (z >= epidermis_thickness) {
                    mua = derm_mua; mus = derm_mus;
                    g = derm_g;     albedo = derm_albedo;
                } else {
                    mua = epi_mua;  mus = epi_mus;
                    g = epi_g;      albedo = epi_albedo;
                }

                // Continue remaining path in new layer
                if (s_left > 1e-12) {
                    r += s_left * ur;
                    z += s_left * uz;
                }

            } else {
                // ── Normal step, no boundary crossed ────────────────────
                r += s * ur;
                z += s * uz;
            }

            // ── Absorption: discrete analog method ───────────────────────
            // Step sampled from mu_t = mua + mus; at each collision event
            // the photon survives with probability albedo = mus/mu_t.
            W *= albedo;

            // ── Henyey-Greenstein scattering (2D form per paper) ─────────
            // In 2D, the phase function gives a deflection angle in the plane.
            double costheta_s;
            rnd = dis(gen);
            if (g == 0.0) {
                costheta_s = 2.0 * rnd - 1.0;
            } else if (g >= COSZERO) {
                costheta_s = 1.0;
            } else {
                double temp = (1.0 - g * g) / (1.0 - g + 2.0 * g * rnd);
                costheta_s  = (1.0 + g * g - temp * temp) / (2.0 * g);
            }
            costheta_s = fmax(-1.0, fmin(1.0, costheta_s));
            double sintheta_s = sqrt(1.0 - costheta_s * costheta_s);

            // 2D direction update: rotate (ur, uz) by deflection angle
            double new_ur =  ur * costheta_s + uz * sintheta_s;
            double new_uz = -ur * sintheta_s + uz * costheta_s;
            ur = new_ur;
            uz = new_uz;

            // ── Russian roulette ─────────────────────────────────────────
            if (W < THRESHOLD) {
                if (dis(gen) <= CHANCE) W /= CHANCE;
                else goto next_photon;
            }
        }

        next_photon:;
    }

    return total_reflection / totalPhotons;
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
    int step_size = 10;
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
        // CONSTANTS — outside the loop
        // ============================================================
        const double p_carotene_epi  = 2.1e-4;   // g/L
        const double p_carotene_derm = 7.0e-5;   // g/L
        const double w_carotene      = 536.87;   // g/mol
        const double pbil            = 0.05;     // g/L
        const double wbil            = 584.66;   // g/mol

        // ============================================================
        // INSIDE WAVELENGTH LOOP
        // ============================================================

        // Index — declared once, used for both tables
        int idx = (nm - 380) / 5;

        // Baseline absorption (paper formula)
        double alpha_base = 7.84e8 * std::pow(nm, -3.255);

        // Melanin
        double alpha_eumelanin   = 6.6e10 * std::pow(nm, -3.33);
        double alpha_pheomelanin = 2.9e14 * std::pow(nm, -4.75);

        // Beta-carotene (physically correct)
        double epsilon_car         = epsilon_betacarotene[idx];
        double alpha_carotene_epi  = epsilon_car * (p_carotene_epi  / w_carotene);
        double alpha_carotene_derm = epsilon_car * (p_carotene_derm / w_carotene);

        // Bilirubin
        double epsilon_bil     = epsilon_bilirubin[idx];
        double alpha_bilirubin = epsilon_bil * (pbil / wbil);

        // Hemoglobin
        auto hb_coefficients = calculate_absorption_coefficient(nm);
        double alpha_HbO2 = hb_coefficients.first;
        double alpha_Hb   = hb_coefficients.second;

        // ============================================================
        // EPIDERMIS ABSORPTION
        // ============================================================
        double melanin_absorption = melanin_blend * alpha_eumelanin
                                + (1.0 - melanin_blend) * alpha_pheomelanin;
        double Uepidermis = melanin_concentration * melanin_absorption
                        + (1.0 - melanin_concentration) * (alpha_base + alpha_carotene_epi);

        // ============================================================
        // DERMIS ABSORPTION
        // ============================================================
        double blood_absorption = blood_oxy * alpha_HbO2
                                + (1.0 - blood_oxy) * alpha_Hb;
        double Udermis = blood_concentration * (blood_absorption + alpha_bilirubin + alpha_carotene_derm)
                    + (1.0 - blood_concentration) * alpha_base;

        // ============================================================
        // SCATTERING
        // ============================================================
        double lambda_normalized = nm / 500.0;
        double rayleigh_term = 0.48 * std::pow(lambda_normalized, -4.0);
        double mie_term      = 0.52 * std::pow(lambda_normalized, -0.22);
        double Us_epidermis  = 36.4 * (rayleigh_term + mie_term);
        double Us_dermis     = Us_epidermis;          // paper uses same for both layers
        double g             = 0.62 + nm * 0.29e-3;  // paper's exact form

        // ============================================================
        // MONTE CARLO LIGHT TRANSPORT  (3-layer: SC → epidermis → dermis)
        // ============================================================
        // T: epidermis thickness in cm (not including SC)

        double reflectance = MonteCarlo(Uepidermis, Us_epidermis, g,
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



void ProcessAndWriteBioSkin(
    double melanin_concentration,
    double blood_concentration,
    double melanin_blend,
    double blood_oxy,
    double epidermis_thickness
) {
    std::vector<double> row = Bioskin(
        melanin_concentration,
        blood_concentration,
        melanin_blend,
        blood_oxy,
        epidermis_thickness
    );

    if (row.empty()) return;

    // ✅ Push to queue (FAST, minimal lock)
    {
        std::lock_guard<std::mutex> lock(write_mtx);
        writeQueue.push(std::move(row));
    }
    write_cv.notify_one();

    // ✅ Progress update
    int done = ++completedTasks;
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

void writerThread(std::ofstream& outputFile) {
    while (true) {
        std::unique_lock<std::mutex> lock(write_mtx);
        write_cv.wait(lock, [] {
            return !writeQueue.empty() || writingFinished;
        });
        while (!writeQueue.empty()) {
            auto row = std::move(writeQueue.front());
            writeQueue.pop();
            lock.unlock();
            WriteRowToCSV(outputFile, row);
            lock.lock();
        }
        if (writingFinished && writeQueue.empty()) break;
    }
}


int main() {
    int numSamples = 15;

    std::vector<double> CmValues = generateSequence(0.001, 0.50, 51, 3);
    std::vector<double> ChValues = generateSequence(0.01, 0.80, 51, 4);
    std::vector<double> BmValues = generateSequence(0.0, 1.0, 5, 1);
    std::vector<double> BloodOxyValues = generateSequence(0.60, 0.98, 13, 1);
    std::vector<double> TValues = generateSequence(0.005, 0.020, 5, 1);

    // ✅ Total tasks
    totalTasks = CmValues.size() * ChValues.size() *
                 BmValues.size() * BloodOxyValues.size() *
                 TValues.size();

    std::cout << "Total datasets: " << totalTasks << std::endl;

    // === File ===
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");

    std::string outputFilename = "lut_rgb_" + ss.str() + ".csv";
    std::ofstream outputFile(outputFilename);

    WriteHeaderToCSVBioWithSpectral(outputFile);

    auto start = std::chrono::high_resolution_clock::now();

    // === Start writer thread ===

    std::thread writer(writerThread, std::ref(outputFile));

    std::atomic<bool> progressRunning(true);
    auto startTime = std::chrono::steady_clock::now();

    std::thread progressThread([&progressRunning, startTime]() {
        using namespace std::chrono;
            while (progressRunning.load()) {
                std::this_thread::sleep_for(minutes(5));
                if (!progressRunning.load()) break;

                int done   = completedTasks.load();
                double pct = 100.0 * done / totalTasks;
                auto now   = steady_clock::now();
                double elapsed = duration<double>(now - startTime).count();
                double eta = (done > 0) ? (elapsed / done) * (totalTasks - done) : 0.0;

                std::cout << "[" << std::fixed << std::setprecision(1) << pct << "%] "
                        << done << " / " << totalTasks
                        << "  elapsed: " << (int)(elapsed/60) << "m" << (int)fmod(elapsed,60) << "s"
                        << "  ETA: "    << (int)(eta/60)     << "m" << (int)fmod(eta,60)    << "s"
                        << "\n";
                std::cout.flush();
            }
        });

    // === Worker threads ===
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    for (int i = 0; i < numThreads; i++)
        workers.emplace_back(worker);

    // === Push tasks ===
    for (auto cm : CmValues)
      for (auto ch : ChValues)
        for (auto bm : BmValues)
          for (auto blood_oxy : BloodOxyValues)
            for (auto t : TValues) {
                {
                    std::lock_guard<std::mutex> lock(task_mtx);
                    tasks.push([cm, ch, bm, blood_oxy, t]() {
                        ProcessAndWriteBioSkin(cm, ch, bm, blood_oxy, t);
                    });
                }
                cv.notify_one();
            }

    // === Signal workers done ===
    { std::lock_guard<std::mutex> lock(task_mtx); finished = true; }
    cv.notify_all();
    for (auto& w : workers) w.join();

    // === Stop progress thread ===
    progressRunning.store(false);
    progressThread.join();

    // === Finish writer ===
    { std::lock_guard<std::mutex> lock(write_mtx); writingFinished = true; }
    write_cv.notify_all();
    writer.join();

    outputFile.close();
    // ... print elapsed time ...
    return 0;
}












































































// int main() {
//     double step_size = 5;
//     int numSamples = 15;


//     //Generate parameter ranges

//     std::vector<double> CmValues = generateSequence(0.05, 0.50, 51, 2);      // 1% to 50%
//     std::vector<double> ChValues = generateSequence(0.02, 0.20, 51, 2);      // 2% to 20% (raised min to ensure haemoglobin features are visible)
//     std::vector<double> BmValues = generateSequence(0.0, 1.0, 5, 2);         // 50% to 100%
//     std::vector<double> BloodOxyValues = generateSequence(0.60, 0.98, 13, 1); // 60% to 98%
//     std::vector<double> TValues = generateSequence(0.005, 0.020, 5, 1);      // 50μm to 200μm



//     totalTasks =  CmValues.size() * ChValues.size() * BmValues.size() *
//         BloodOxyValues.size() * TValues.size();

//     // Get current time
//     auto now = std::chrono::system_clock::now();
//     std::time_t now_c = std::chrono::system_clock::to_time_t(now);

//     // Format datetime
//     std::stringstream ss;
//     ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");

//     std::string outputFilename = "lut_rgb_"+ ss.str() + ".csv";
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

// double MonteCarlo(
//                   double epi_mua, double epi_mus, double epi_g,
//                   double derm_mua, double derm_mus, double derm_g,
//                   double epidermis_thickness) {
//     int Nphotons = 1000000;
//     double epi_albedo  = epi_mus / (epi_mus + epi_mua);
//     double derm_albedo = derm_mus / (derm_mus + derm_mua);
//     int NR = Nbins; //number of radial bins
//     double radial_size = 2.5;
//     double r = 0.0;
//     // int ir = 0;
//     double dr = radial_size / NR;
//     //random seed
//     std::vector<double> ReflBin(NR + 1, 0.0);
//     srand(time(NULL));
//     for (int i = 0; i < Nbinsp1; i++) {
//         ReflBin[i] = 0;
//     }

//     for (int i_photon = 0; i_photon < Nphotons; i_photon++) {
//         double W = 1.0;
//         int photon_status = ALIVE;
//         double x = 0.0;
//         double y = 0.0;
//         double z = 0.0;
//         double ux, uy, uz;
//         double costheta, sintheta, cospsi, sinpsi, psi, uxx, uyy, uzz;
//         double s, rnd;
//         int it, ir;
//         double mua = epi_mua;
//         double mus = epi_mus;
//         double albedo = epi_albedo;
//         double absorb;

//         // Randomly set photon trajectory to yield an isotropic source.
//         costheta = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1.0;
//         sintheta = sqrt(1.0 - costheta * costheta);
//         psi = 2.0 * PI * static_cast<double>(rand()) / RAND_MAX;
//         // std::cout << "psi: " << psi << std::endl;
//         ux = sintheta * cos(psi);
//         uy = sintheta * sin(psi);
//         uz = (fabs(costheta)); // fabs is

//         // Propagate one photon until it dies as determined by ROULETTE or reaches the surface
//         it = 0;
//         const int max_iterations = 10000;
//         while (true) {
//             it++;
//             rnd = static_cast<double>(rand()) / RAND_MAX;
//             // std::cout << "rnd: " << rnd << std::endl;
//             while (rnd <= 0.0) {
//                 rnd = static_cast<double>(rand()) / RAND_MAX;

//             }
//             s = -log(rnd) / (mua + mus);
//             x = x + (s * ux);
//             y = y + (s * uy);
//             z = z + (s * uz);

//             if (uz < 0) {
//                 // --- FIX 2: Recover pre-step position, then step exactly to z=0 ---
//                 double x_old = x - s * ux;
//                 double y_old = y - s * uy;
//                 double z_old = z - s * uz;  // z_old > 0 (photon was inside tissue)

//                 // Correct path length from pre-step position to the surface (z=0)
//                 double s1 = z_old / (-uz);  // uz < 0, so -uz > 0; s1 > 0

//                 // Exact surface position
//                 double x_surf = x_old + s1 * ux;
//                 double y_surf = y_old + s1 * uy;
//                 // z at surface = 0

//                 // --- FIX 1: Fresnel at tissue->air interface (nt=1.33 -> 1.0) ---
//                 double internal_reflectance = RFresnel(nt, 1.0, -uz);
//                 double external_reflectance = 1.0 - internal_reflectance;

//                 r = sqrt(x_surf * x_surf + y_surf * y_surf);
//                 ir = static_cast<int>(r / dr);
//                 if (ir >= NR) {
//                     ir = NR;
//                 }
//                 if (ir < 0) {
//                     ir = 0;
//                 }
//                 ReflBin[ir] = ReflBin[ir] + (W * external_reflectance);
//                 W = internal_reflectance * W;

//                 // Reflect direction and continue remaining path from the surface
//                 uz = -uz;  // now positive (heading back into tissue)
//                 double s_remaining = s - s1;
//                 x = x_surf + s_remaining * ux;
//                 y = y_surf + s_remaining * uy;
//                 z = s_remaining * uz;  // z_surface = 0, so z = s_remaining * uz
//             }

//             if (z < epidermis_thickness) {
//                 mua = epi_mua;
//                 mus = epi_mus;
//                 albedo = epi_albedo;
//             }
//             else {
//                 mua = derm_mua;
//                 mus = derm_mus;
//                 albedo = derm_albedo;
//             }

//             absorb = W * (1 - albedo);
//             W = W - absorb;

//             // Determine which g to use based on layer
//             double current_g;
//             if (z < epidermis_thickness) {
//                 current_g = epi_g;
//             } else {
//                 current_g = derm_g;
//             }

//             // Sample for costheta
//             rnd = static_cast<double>(rand()) / RAND_MAX;
//             if (current_g == 0.0) {
//                 costheta = 2.0 * rnd - 1.0;
//             }
//             else {
//                 double temp = (1.0 - current_g * current_g) / (1.0 - current_g + 2 * current_g * rnd);
//                 costheta = (1.0 + current_g * current_g - temp * temp) / (2.0 * current_g);
//             }
//             sintheta = sqrt(1.0 - costheta * costheta);

//             // Sample psi
//             psi = 2.0 * PI * static_cast<double>(rand()) / RAND_MAX;
//             cospsi = cos(psi);
//             if (psi < PI) {
//                 sinpsi = sqrt(1.0 - cospsi * cospsi);
//             }
//             else {
//                 sinpsi = -sqrt(1.0 - cospsi * cospsi);
//             }

//             if (1 - abs(uz) <= ONE_MINUS_COSZERO) {
//                 uxx = sintheta * cospsi;
//                 uyy = sintheta * sinpsi;
//                 uzz = costheta * copysign(uz, -1.0);
//             }
//             else {
//                 double temp = sqrt(1.0 - uz * uz);
//                 uxx = sintheta * (ux * uz * cospsi - uy * sinpsi) / temp + ux * costheta;
//                 uyy = sintheta * (uy * uz * cospsi + ux * sinpsi) / temp + uy * costheta;
//                 uzz = -sintheta * cospsi * temp + uz * costheta;
//             }
//             //update trajectory
//             ux = uxx;
//             uy = uyy;
//             uz = uzz;
//             if (W < THRESHOLD) {
//                 if (static_cast<double>(rand()) / RAND_MAX <= CHANCE) {
//                     W = W / CHANCE;
//                 }
//                 else {
//                     photon_status = DEAD;
//                 }
//             }
//             if (photon_status == DEAD || it > max_iterations) {
//                 break;
//             }
//         }
//         if (i_photon >= Nphotons) {
//             break;
//         }
//     }
//     double total_reflection = 0.0;
//     for (int i = 0; i < NR; i++) {
//     total_reflection += ReflBin[i];
//     }
//     return total_reflection / Nphotons;
// }




















