// ============================================================
// bioskin_gpu.cu  —  Full CUDA port of BioSkin Monte Carlo LUT
// Target: NVIDIA RTX 5070 Ti  (Ada Lovelace, sm_89)
//
// Build:
//   nvcc -O3 -arch=sm_89 -o bioskin_gpu bioskin_gpu.cu -lcurand
//
// For other GPUs, change -arch:
//   RTX 3000/4000 series  → -arch=sm_86
//   RTX 2000 series       → -arch=sm_75
// ============================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <atomic>

// ============================================================
// CUDA ERROR CHECKING
// ============================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================
// CONSTANTS  (must be visible to both host and device)
// ============================================================
#define NBINS       1000
#define NBINSP1     1001
#define PI_VAL      3.1415926f
#define ALIVE       1
#define DEAD        0
#define THRESHOLD   0.01f
#define CHANCE      0.1f
#define COS90D      1e-6f
#define ONE_MINUS_COSZERO 1e-12f
#define COSZERO     (1.0f - 1.0e-12f)
#define NT          1.33f
#define NPHOTONS    10000
#define N_WAVELENGTHS 85          // 380–800 nm in 5 nm steps
#define STEP_NM     5

// ============================================================
// SPECTRAL DATA  (constant memory — very fast on GPU)
// ============================================================

// Haemoglobin data (81 entries covering 380–780 in 5 nm steps)
// deoxy = HbO2 in original code (first of the pair)
// oxy   = Hb   in original code (second of the pair)
__constant__ float c_deoxy[81] = {
    97.58f,97.24668630103302f,100.024f,105.91194109690093f,119.18599999999998f,
    144.12166711756137f,171.316f,191.36605619773087f,213.689f,247.70199569605347f,
    272.27f,266.2579696259483f,223.01f,135.87018654825664f,51.604f,
    16.97691108451183f,9.882f,8.212346944672365f,8.899f,8.873007247453975f,
    8.634f,8.681609570603786f,9.119f,10.049335328923313f,11.365f,
    12.958378455856344f,14.673f,16.35239393593862f,18.105f,20.039257928511944f,
    22.129f,24.348058492989708f,26.427f,28.096391113549814f,29.245f,
    29.76159482571142f,29.669f,28.990039932181688f,27.538f,25.126165581198457f,
    21.955f,18.22496658062756f,14.467f,11.212034935036193f,8.525f,
    6.470823809155298f,4.979f,3.9790222100320287f,3.337f,2.919042930652537f,
    2.646f,2.4387202060527526f,2.276f,2.1366358330309474f,2.015f,
    1.905464795761565f,1.803f,1.702575392399664f,1.601f,1.4950828498404518f,
    1.386f,1.2749275085576262f,1.167f,1.0673520988137917f,1.0f,
    0.9889598985596247f,0.999f,0.9948885098284603f,0.987f,0.9857090424696132f,
    0.985f,0.9788572353538606f,0.97f,0.9611475454072232f,0.95f,
    0.9342574922028001f,0.92f,0.9133075013759765f,0.91f,0.9058974995413411f,0.901f
};

__constant__ float c_oxy[81] = {
    185.0f,192.57373720944835f,195.0f,192.2787883716549f,200.0f,
    233.75353256062215f,270.473f,287.09201626461225f,263.925f,181.28636985170434f,
    108.075f,113.18976462516179f,131.344f,97.25104239732498f,52.191f,
    37.443980990888335f,36.029f,30.965071657345057f,24.951f,20.68558906504135f,
    17.759f,15.761393952406863f,14.372f,13.27004722051748f,12.369f,
    11.582322724488264f,11.005f,10.732016432552953f,10.893f,11.617578680194026f,
    13.921f,18.8185114862829f,24.401f,28.759352402108576f,30.281f,
    27.353374101065647f,22.739f,19.200402991497548f,17.885f,19.940207949949095f,
    23.322f,25.986349308807895f,24.521f,15.513696197203544f,5.762f,
    2.0634735079708535f,1.475f,1.0534627549713358f,0.778f,0.6277499622011322f,
    0.53f,0.4120374718218712f,0.307f,0.24802520686764065f,0.219f,
    0.2038112869722848f,0.196f,0.18910707129865068f,0.182f,0.17354628523581117f,
    0.165f,0.1576152172864823f,0.152f,0.1487624110452951f,0.148f,
    0.14981031644174703f,0.152f,0.1523756903042227f,0.152f,0.15193554173291662f,
    0.152f,0.15201105929827746f,0.152f,0.15199810247741857f,0.152f,
    0.15200032583721096f,0.152f,0.1519999424993157f,0.152f,0.15200001916689476f,0.152f
};

// D65 illuminant — 85 entries (380–800 nm in 5 nm steps)
__constant__ float c_d65[85] = {
    82.7549f,91.486f,93.4318f,86.6823f,104.865f,
    117.008f,117.812f,114.861f,115.923f,108.811f,
    109.354f,107.802f,104.790f,107.689f,104.405f,
    104.046f,100.0f,96.3342f,95.788f,97.7018f,
    98.9178f,95.6748f,96.7282f,101.898f,100.888f,
    102.074f,100.0f,97.3083f,97.0288f,98.1618f,
    100.026f,98.0228f,98.1671f,101.891f,98.7243f,
    100.413f,103.168f,101.013f,100.88f,102.116f,
    101.023f,98.6476f,99.4571f,98.0009f,97.9567f,
    99.1716f,100.655f,100.198f,101.009f,101.588f,
    101.438f,100.65f,100.0f,99.9955f,100.015f,
    99.6763f,99.3541f,99.1406f,99.0029f,98.7686f,
    98.3861f,97.7786f,97.1823f,96.8571f,96.6482f,
    96.4405f,96.4143f,97.1398f,97.6742f,96.0826f,
    95.5264f,96.6574f,98.5631f,96.6453f,95.0694f,
    97.0668f,97.9226f,96.6747f,95.0018f,94.7468f,
    97.1428f,80.2146f,81.2462f,69.7213f,71.6091f
};

// ============================================================
// DEVICE FUNCTIONS
// ============================================================

// --- Absorption coefficient lookup (linear interp, 5 nm grid) ---
__device__ void calc_hb(int nm, float* hbo2_out, float* hb_out)
{
    // Grid: 380, 385, ..., 780  →  index = (nm - 380) / 5
    int idx = (nm - 380) / 5;
    if (idx < 0)  idx = 0;
    if (idx > 80) idx = 80;
    *hbo2_out = c_deoxy[idx];   // note: deoxy array = HbO2 per original code
    *hb_out   = c_oxy[idx];
}

// --- D65 value (85-point grid, 380–800 nm) ---
__device__ float d65_val(int nm)
{
    int idx = (nm - 380) / 5;
    if (idx < 0)  idx = 0;
    if (idx > 84) idx = 84;
    return c_d65[idx];
}

// --- CIE 1931 colour matching functions ---
__device__ float xFit(float w)
{
    float t1 = (w-442.0f)*((w<442.0f)?0.0624f:0.0374f);
    float t2 = (w-599.8f)*((w<599.8f)?0.0264f:0.0323f);
    float t3 = (w-501.1f)*((w<501.1f)?0.0490f:0.0382f);
    return 0.362f*expf(-0.5f*t1*t1)+1.056f*expf(-0.5f*t2*t2)-0.065f*expf(-0.5f*t3*t3);
}
__device__ float yFit(float w)
{
    float t1 = (w-568.8f)*((w<568.8f)?0.0213f:0.0247f);
    float t2 = (w-530.9f)*((w<530.9f)?0.0613f:0.0322f);
    return 0.821f*expf(-0.5f*t1*t1)+0.286f*expf(-0.5f*t2*t2);
}
__device__ float zFit(float w)
{
    float t1 = (w-437.0f)*((w<437.0f)?0.0845f:0.0278f);
    float t2 = (w-459.0f)*((w<459.0f)?0.0385f:0.0725f);
    return 1.217f*expf(-0.5f*t1*t1)+0.681f*expf(-0.5f*t2*t2);
}

// --- Fresnel reflection coefficient ---
__device__ float RFresnel_d(float n1, float n2, float cosT1)
{
    if (n1 == n2) return 0.0f;
    if (cosT1 > COSZERO) { float r=(n2-n1)/(n2+n1); return r*r; }
    if (cosT1 < COS90D)  return 1.0f;
    float sinT1 = sqrtf(1.0f - cosT1*cosT1);
    float sinT2 = n1*sinT1/n2;
    if (sinT2 >= 1.0f) return 1.0f;
    float cosT2 = sqrtf(1.0f - sinT2*sinT2);
    float cosAP = cosT1*cosT2 - sinT1*sinT2;
    float cosAM = cosT1*cosT2 + sinT1*sinT2;
    float sinAP = sinT1*cosT2 + cosT1*sinT2;
    float sinAM = sinT1*cosT2 - cosT1*sinT2;
    return 0.5f*sinAM*sinAM*(cosAM*cosAM+cosAP*cosAP)/(sinAP*sinAP*cosAM*cosAM);
}

// --- Gamma correction (sRGB) ---
__device__ float gamma_corr(float C)
{
    float a = fabsf(C);
    return (a > 0.0031308f) ? (1.055f*powf(a,1.0f/2.4f)-0.055f) : (12.92f*C);
}

// ============================================================
// MONTE CARLO KERNEL
// Each thread handles ONE (parameter_set × wavelength) pair.
// Grid layout:
//   blockDim.x = BLOCK_WL  (wavelengths per block, e.g. 85)
//   gridDim.x  = total_param_combos
// So thread (bx, tx) processes parameter[bx] at wavelength[tx].
// ============================================================

// Output layout per parameter combination (one row):
//   [0..4]    = Cm, Ch, Bm, BloodOxy, T
//   [5..89]   = reflectances[0..84]
//   [90..92]  = X, Y, Z
//   [93..95]  = sR, sG, sB
//   [96]      = validity flag (1.0 = valid, 0.0 = skip)
#define ROW_FLOATS  97
#define FLAG_IDX    96

__global__ void bioskin_kernel(
    const float* __restrict__ d_params,   // [N_combos × 5]
    float*       __restrict__ d_output,   // [N_combos × ROW_FLOATS]
    int          N_combos,
    unsigned long long seed)
{
    int combo_idx = blockIdx.x;
    int wl_idx    = threadIdx.x;   // 0..84 → wavelength 380..800

    if (combo_idx >= N_combos) return;
    if (wl_idx    >= N_WAVELENGTHS) return;

    // --- Load parameters ---
    float Cm  = d_params[combo_idx*5 + 0];
    float Ch  = d_params[combo_idx*5 + 1];
    float Bm  = d_params[combo_idx*5 + 2];
    float Boxy= d_params[combo_idx*5 + 3];
    float T   = d_params[combo_idx*5 + 4];

    int nm = 380 + wl_idx * STEP_NM;   // wavelength in nm

    // --- Optical coefficients ---
    float alpha_base       = 0.0244f + 8.53f*expf(-(nm-154.0f)/66.2f);
    float alpha_eumela     = 6.6e10f * powf((float)nm, -3.33f);
    float alpha_pheomela   = 2.9e14f * powf((float)nm, -4.75f);
    float alpha_car_epi    = 2.1e-4f;
    float alpha_car_derm   = 7.0e-5f;

    float hbo2, hb;
    calc_hb(nm, &hbo2, &hb);

    float melanin_abs = Bm*alpha_eumela + (1.0f-Bm)*alpha_pheomela;
    float mua_epi     = Cm*melanin_abs + (1.0f-Cm)*(alpha_base+alpha_car_epi);
    float blood_abs   = Boxy*hbo2 + (1.0f-Boxy)*hb;
    float mua_derm    = Ch*(blood_abs+alpha_car_derm) + (1.0f-Ch)*alpha_base;

    float lam_n      = nm/500.0f;
    float mus_epi    = 36.4f*(0.48f*powf(lam_n,-4.0f)+0.52f*powf(lam_n,-0.22f));
    float mus_derm   = 0.65f*mus_epi;
    float g          = 0.62f + 0.00029f*(nm-380.0f);

    // Stratum corneum
    const float sc_thickness = 0.0015f;
    float sc_mua = alpha_base;
    float sc_mus = 100.0f*powf(400.0f/nm, 0.8f);
    const float sc_g = 0.70f;

    float epi_bottom = sc_thickness + T;

    // --- Monte Carlo per wavelength ---
    float sc_alb   = sc_mus  / (sc_mus  + sc_mua);
    float epi_alb  = mus_epi / (mus_epi + mua_epi);
    float derm_alb = mus_derm/ (mus_derm + mua_derm);

    float radial_size = 2.5f;
    float dr = radial_size / NBINS;

    // Per-thread RNG state
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)combo_idx*N_WAVELENGTHS + wl_idx, 0, &rng);

    float ReflBin[NBINS+1];
    for (int i = 0; i <= NBINS; i++) ReflBin[i] = 0.0f;

    for (int i_photon = 0; i_photon < NPHOTONS; i_photon++) {
        float W  = 1.0f;
        int   ps = ALIVE;
        float x=0,y=0,z=0,ux,uy,uz;
        float costheta, sintheta, cospsi, sinpsi, psi;

        // Initial direction (isotropic source)
        costheta = 2.0f*curand_uniform(&rng) - 1.0f;
        sintheta = sqrtf(fmaxf(0.0f, 1.0f-costheta*costheta));
        psi      = 2.0f*PI_VAL*curand_uniform(&rng);
        ux = sintheta*cosf(psi);
        uy = sintheta*sinf(psi);
        uz = fabsf(costheta);

        float mua=sc_mua, mus=sc_mus, albedo=sc_alb;

        for (int it = 0; it < 10000; it++) {
            float rnd = curand_uniform(&rng);
            while (rnd <= 0.0f) rnd = curand_uniform(&rng);
            float s = -logf(rnd)/(mua+mus);
            x += s*ux; y += s*uy; z += s*uz;

            if (uz < 0) {
                float xo = x-s*ux, yo = y-s*uy, zo = z-s*uz;
                float s1 = zo/(-uz);
                float xs = xo+s1*ux, ys = yo+s1*uy;
                float ext_r = 1.0f - RFresnel_d(NT, 1.0f, -uz);
                float r2 = sqrtf(xs*xs+ys*ys);
                int ir = (int)(r2/dr);
                if (ir >= NBINS) ir = NBINS;
                if (ir < 0)      ir = 0;
                ReflBin[ir] += W*ext_r;
                W *= RFresnel_d(NT, 1.0f, -uz);
                uz = -uz;
                float sr = s-s1;
                x = xs+sr*ux; y = ys+sr*uy; z = sr*uz;
            }

            // Layer selection
            if      (z < sc_thickness) { mua=sc_mua;  mus=sc_mus;  albedo=sc_alb;   }
            else if (z < epi_bottom)   { mua=mua_epi; mus=mus_epi; albedo=epi_alb;  }
            else                       { mua=mua_derm;mus=mus_derm;albedo=derm_alb; }

            W -= W*(1.0f-albedo);

            float cur_g;
            if      (z < sc_thickness) cur_g = sc_g;
            else if (z < epi_bottom)   cur_g = g;
            else                       cur_g = g;

            rnd = curand_uniform(&rng);
            if (cur_g == 0.0f) {
                costheta = 2.0f*rnd-1.0f;
            } else {
                float tmp = (1.0f-cur_g*cur_g)/(1.0f-cur_g+2.0f*cur_g*rnd);
                costheta  = (1.0f+cur_g*cur_g-tmp*tmp)/(2.0f*cur_g);
            }
            sintheta = sqrtf(fmaxf(0.0f,1.0f-costheta*costheta));

            psi    = 2.0f*PI_VAL*curand_uniform(&rng);
            cospsi = cosf(psi);
            sinpsi = (psi<PI_VAL) ? sqrtf(fmaxf(0.0f,1.0f-cospsi*cospsi))
                                  : -sqrtf(fmaxf(0.0f,1.0f-cospsi*cospsi));

            float uxx,uyy,uzz;
            if (1.0f-fabsf(uz) <= ONE_MINUS_COSZERO) {
                uxx = sintheta*cospsi;
                uyy = sintheta*sinpsi;
                uzz = costheta*((uz>=0)?1.0f:-1.0f)*(-1.0f);
            } else {
                float tmp2 = sqrtf(fmaxf(0.0f,1.0f-uz*uz));
                uxx = sintheta*(ux*uz*cospsi-uy*sinpsi)/tmp2 + ux*costheta;
                uyy = sintheta*(uy*uz*cospsi+ux*sinpsi)/tmp2 + uy*costheta;
                uzz = -sintheta*cospsi*tmp2 + uz*costheta;
            }
            ux=uxx; uy=uyy; uz=uzz;

            if (W < THRESHOLD) {
                if (curand_uniform(&rng) <= CHANCE) W /= CHANCE;
                else { ps = DEAD; break; }
            }
            if (ps == DEAD) break;
        }
    }

    float total_refl = 0.0f;
    for (int i = 0; i < NBINS; i++) total_refl += ReflBin[i];
    float reflectance = total_refl / NPHOTONS;

    // Write reflectance into shared output (safe: each wl_idx writes different column)
    float* row = d_output + (long long)combo_idx * ROW_FLOATS;
    row[5 + wl_idx] = reflectance;

    // --- One thread per combo does the XYZ+RGB accumulation after sync ---
    // We use __syncthreads() so wl_idx=0 reads all reflectances
    __syncthreads();

    if (wl_idx == 0) {
        // Write params
        row[0] = Cm;  row[1] = Ch;  row[2] = Bm;
        row[3] = Boxy; row[4] = T;

        // Compute XYZ normalisation
        float norm = 0.0f;
        float X=0,Y=0,Z=0;
        for (int wi = 0; wi < N_WAVELENGTHS; wi++) {
            float w    = 380.0f + wi*STEP_NM;
            float y_cmf= yFit(w);
            float d65  = d65_val((int)w);
            norm += y_cmf * d65 * STEP_NM;
        }
        for (int wi = 0; wi < N_WAVELENGTHS; wi++) {
            float w   = 380.0f + wi*STEP_NM;
            float refl= row[5+wi];
            float d65 = d65_val((int)w);
            X += xFit(w)*refl*d65*STEP_NM;
            Y += yFit(w)*refl*d65*STEP_NM;
            Z += zFit(w)*refl*d65*STEP_NM;
        }
        X = 100.0f*X/norm;
        Y = 100.0f*Y/norm;
        Z = 100.0f*Z/norm;
        row[90]=X; row[91]=Y; row[92]=Z;

        // XYZ → sRGB
        float xn=X/100.0f, yn=Y/100.0f, zn=Z/100.0f;
        float r = xn* 3.2406f + yn*(-1.5372f) + zn*(-0.4986f);
        float g2= xn*(-0.9689f)+ yn* 1.8758f  + zn* 0.0415f;
        float b = xn* 0.0557f  + yn*(-0.204f)  + zn* 1.057f;
        r = gamma_corr(r)*255.0f;
        g2= gamma_corr(g2)*255.0f;
        b = gamma_corr(b)*255.0f;
        // Round to 3dp
        r  = roundf(r *1000.0f)/1000.0f;
        g2 = roundf(g2*1000.0f)/1000.0f;
        b  = roundf(b *1000.0f)/1000.0f;
        row[93]=r; row[94]=g2; row[95]=b;

        // Validity flag
        row[FLAG_IDX] = (r>255||g2>255||b>255) ? 0.0f : 1.0f;
    }
}

// ============================================================
// HOST UTILITY — generate parameter sequence
// ============================================================
static std::vector<double> genSeq(double start, double end, int n, double exp_)
{
    std::vector<double> v(n);
    for (int i=0;i<n;i++) {
        double u = (n>1) ? (double)i/(n-1) : 0.0;
        v[i] = start + (end-start)*pow(u,exp_);
    }
    return v;
}

// ============================================================
// PROGRESS HELPER
// ============================================================
static long long g_total_combos = 0;
static std::chrono::steady_clock::time_point g_t0;

static void printProgress(long long done)
{
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-g_t0).count();
    double pct = 100.0*done/g_total_combos;
    double eta = (done>0 && done<g_total_combos)
                 ? elapsed*(g_total_combos-done)/done : 0.0;
    int em=(int)elapsed/60, es=(int)elapsed%60;
    int am=(int)eta/60,     as_=(int)eta%60;
    printf("[Progress] %lld / %lld  (%.1f%%)  elapsed: %dm%ds  ETA: %dm%ds\n",
           done,g_total_combos,pct,em,es,am,as_);
    fflush(stdout);
}

// ============================================================
// MAIN
// ============================================================
int main()
{
    // --- GPU info ---
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d, %d SMs, %.1f GB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem/1e9);

    // --- Parameter grid ---
    auto Cm_v   = genSeq(0.01, 0.50, 20, 2);
    auto Ch_v   = genSeq(0.01, 0.32, 20, 2);
    auto Bm_v   = genSeq(0.0,  1.0,   5, 2);
    auto Boxy_v = genSeq(0.60, 0.98, 10, 1);
    auto T_v    = genSeq(0.005,0.020,  3, 1);

    long long N = (long long)Cm_v.size()*Ch_v.size()*Bm_v.size()*Boxy_v.size()*T_v.size();
    g_total_combos = N;
    printf("Total parameter combinations: %lld\n", N);
    printf("Progress printed every 5000 rows.\n\n");

    // --- Build flat params array ---
    std::vector<float> h_params(N*5);
    long long idx=0;
    for (double cm  : Cm_v)
    for (double ch  : Ch_v)
    for (double bm  : Bm_v)
    for (double boxy: Boxy_v)
    for (double t   : T_v) {
        h_params[idx*5+0]=(float)cm;
        h_params[idx*5+1]=(float)ch;
        h_params[idx*5+2]=(float)bm;
        h_params[idx*5+3]=(float)boxy;
        h_params[idx*5+4]=(float)t;
        idx++;
    }

    // --- Allocate GPU memory ---
    float *d_params, *d_output;
    size_t params_bytes = N*5*sizeof(float);
    size_t output_bytes = N*(long long)ROW_FLOATS*sizeof(float);

    printf("GPU memory required: %.2f GB\n\n", (params_bytes+output_bytes)/1e9);

    CUDA_CHECK(cudaMalloc(&d_params, params_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));
    CUDA_CHECK(cudaMemcpy(d_params, h_params.data(), params_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output, 0, output_bytes));

    // --- Output file ---
    auto now_t = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now_t);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    std::string filename = "lut_rgb_gpu_" + ss.str() + ".csv";
    std::ofstream outf(filename);

    // Header
    outf << "melanin_concentration(Cm),blood_concentration(Ch),melanin_blend(Bm),"
            "BloodOxy,epidermis_thickness(T),";
    for (int wl=380;wl<=800;wl+=5) outf<<"R_"<<wl<<"nm,";
    outf << "X,Y,Z,sR,sG,sB\n";

    // --- Launch kernel ---
    // Each block = one parameter combo; threads within block = 85 wavelengths
    // We process in batches to allow progress reporting and avoid TDR timeout
    const long long BATCH = 5000;
    unsigned long long seed = (unsigned long long)time(nullptr);

    g_t0 = std::chrono::steady_clock::now();
    long long total_written = 0;

    for (long long batch_start = 0; batch_start < N; batch_start += BATCH)
    {
        long long batch_end = std::min(batch_start + BATCH, N);
        long long batch_n   = batch_end - batch_start;

        // Sub-slice pointers
        float* d_p_batch = d_params + batch_start*5;
        float* d_o_batch = d_output + batch_start*(long long)ROW_FLOATS;

        dim3 grid((unsigned int)batch_n);
        dim3 block(N_WAVELENGTHS);   // 85 threads per block

        bioskin_kernel<<<grid, block>>>(d_p_batch, d_o_batch, (int)batch_n, seed+batch_start);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy batch results to host & write CSV
        std::vector<float> h_out(batch_n*(long long)ROW_FLOATS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_o_batch,
                              batch_n*(long long)ROW_FLOATS*sizeof(float),
                              cudaMemcpyDeviceToHost));

        for (long long r=0;r<batch_n;r++) {
            float* row = h_out.data() + r*(long long)ROW_FLOATS;
            if (row[FLAG_IDX] < 0.5f) continue;  // invalid RGB — skip
            for (int c=0;c<ROW_FLOATS-1;c++) {   // -1 to exclude flag
                outf << row[c];
                if (c < ROW_FLOATS-2) outf << ',';
            }
            outf << '\n';
            total_written++;
        }

        printProgress(batch_end);
    }

    // --- Final summary ---
    double total_elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now()-g_t0).count();
    printf("\n=== Done ===\n");
    printf("Total rows written : %lld / %lld\n", total_written, N);
    printf("Total elapsed time : %.2fs  (%dm %ds)\n",
           total_elapsed, (int)total_elapsed/60, (int)total_elapsed%60);
    printf("Output file        : %s\n", filename.c_str());

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_output));
    outf.close();
    return 0;
}
