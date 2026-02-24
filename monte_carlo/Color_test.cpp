#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Copy your color matching functions here
double xFit_1931(double wave) {
    double t1 = (wave - 442.0) * ((wave < 442.0) ? 0.0624 : 0.0374);
    double t2 = (wave - 599.8) * ((wave < 599.8) ? 0.0264 : 0.0323);
    double t3 = (wave - 501.1) * ((wave < 501.1) ? 0.0490 : 0.0382);
    return 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2)
        - 0.065 * exp(-0.5 * t3 * t3);
}

double yFit_1931(double wave) {
    double t1 = (wave - 568.8) * ((wave < 568.8) ? 0.0213 : 0.0247);
    double t2 = (wave - 530.9) * ((wave < 530.9) ? 0.0613 : 0.0322);
    return 0.821 * exp(-0.5 * t1 * t1) + 0.286 * exp(-0.5 * t2 * t2);
}

double zFit_1931(double wave) {
    double t1 = (wave - 437.0) * ((wave < 437.0) ? 0.0845 : 0.0278);
    double t2 = (wave - 459.0) * ((wave < 459.0) ? 0.0385 : 0.0725);
    return 1.217 * exp(-0.5 * t1 * t1) + 0.681 * exp(-0.5 * t2 * t2);
}

// Simplified D65 - approximate values
double getD65Value(int wavelength) {
    // Simplified D65 approximation for testing
    // Peak around 500-550nm at ~100, drops off at edges
    if (wavelength < 400) return 50.0;
    if (wavelength < 500) return 50.0 + (wavelength - 400) * 0.5;
    if (wavelength <= 560) return 100.0;
    if (wavelength <= 700) return 100.0 - (wavelength - 560) * 0.2;
    return 70.0;
}

double gamma_correction(double C) {
    double abs_C = std::abs(C);
    if (abs_C > 0.0031308) {
        return 1.055 * std::pow(abs_C, 1.0 / 2.4) - 0.055;  // Using 2.4 (correct sRGB)
    }
    else {
        return 12.92 * C;
    }
}

std::vector<double> XYZ_to_sRGB(std::vector<double> xyz) {
    double x = xyz[0];
    double y = xyz[1];
    double z = xyz[2];

    std::cout << "\n=== XYZ to sRGB Conversion ===" << std::endl;
    std::cout << "Input XYZ: (" << x << ", " << y << ", " << z << ")" << std::endl;

    // sRGB transformation matrix
    std::vector<std::vector<double>> mat3x3 = {
        {3.2406, -1.5372, -0.4986},
        {-0.9689, 1.8758, 0.0415},
        {0.0557, -0.204, 1.057}
    };

    double r = x * mat3x3[0][0] + y * mat3x3[0][1] + z * mat3x3[0][2];
    double g = x * mat3x3[1][0] + y * mat3x3[1][1] + z * mat3x3[1][2];
    double b = x * mat3x3[2][0] + y * mat3x3[2][1] + z * mat3x3[2][2];

    std::cout << "Linear RGB (0-1 scale): (" << r << ", " << g << ", " << b << ")" << std::endl;

    // Check if values are in reasonable range
    if (r < 0 || r > 1.5 || g < 0 || g > 1.5 || b < 0 || b > 1.5) {
        std::cout << "WARNING: Linear RGB values outside expected range!" << std::endl;
    }

    r = gamma_correction(r);
    g = gamma_correction(g);
    b = gamma_correction(b);

    std::cout << "After gamma correction (0-1 scale): (" << r << ", " << g << ", " << b << ")" << std::endl;

    r = r * 255.0;
    g = g * 255.0;
    b = b * 255.0;

    std::cout << "Final RGB (0-255 scale): (" << r << ", " << g << ", " << b << ")" << std::endl;

    // Round to 3 decimal places
    r = std::round(r * 1000.0) / 1000.0;
    g = std::round(g * 1000.0) / 1000.0;
    b = std::round(b * 1000.0) / 1000.0;

    std::vector<double> sRGB = {r, g, b};
    return sRGB;
}

void test_perfect_white() {
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          TEST 1: Perfect White Surface                ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    int step_size = 5;
    std::vector<double> total = {0.0, 0.0, 0.0};
    
    // Perfect white: reflectance = 1.0 at all wavelengths
    std::cout << "\nSimulating perfect white (reflectance = 1.0 everywhere)" << std::endl;
    
    for (int nm = 380; nm <= 800; nm += step_size) {
        double reflectance = 1.0;  // Perfect white
        double d65 = getD65Value(nm);
        double x = xFit_1931(nm);
        double y = yFit_1931(nm);
        double z = zFit_1931(nm);
        
        total[0] += x * reflectance * d65 * step_size;
        total[1] += y * reflectance * d65 * step_size;
        total[2] += z * reflectance * d65 * step_size;
    }
    
    std::cout << "Raw XYZ (before normalization): (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    // Compute normalization
    double normalization = 0.0;
    for (int nm = 380; nm <= 800; nm += step_size) {
        double y = yFit_1931(nm);
        double d65 = getD65Value(nm);
        normalization += y * d65 * step_size;
    }
    
    std::cout << "Normalization constant: " << normalization << std::endl;
    
    // Normalize to Y=100 for perfect white
    total[0] = 100.0 * total[0] / normalization;
    total[1] = 100.0 * total[1] / normalization;
    total[2] = 100.0 * total[2] / normalization;
    
    std::cout << "Normalized XYZ (should be ~95, 100, 108): (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    std::vector<double> rgb = XYZ_to_sRGB(total);
    
    std::cout << "\n✓ Expected: RGB ≈ (255, 255, 255) - Pure white" << std::endl;
    std::cout << "✓ Got: RGB = (" << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << ")" << std::endl;
    
    if (std::abs(rgb[0] - 255) < 10 && std::abs(rgb[1] - 255) < 10 && std::abs(rgb[2] - 255) < 10) {
        std::cout << "✓ PASS: White color correct!" << std::endl;
    } else {
        std::cout << "✗ FAIL: White color incorrect!" << std::endl;
    }
}

void test_neutral_gray() {
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          TEST 2: Neutral Gray (50% reflectance)       ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    int step_size = 5;
    std::vector<double> total = {0.0, 0.0, 0.0};
    
    // Neutral gray: reflectance = 0.5 at all wavelengths
    std::cout << "\nSimulating neutral gray (reflectance = 0.5 everywhere)" << std::endl;
    
    for (int nm = 380; nm <= 800; nm += step_size) {
        double reflectance = 0.5;  // 50% gray
        double d65 = getD65Value(nm);
        double x = xFit_1931(nm);
        double y = yFit_1931(nm);
        double z = zFit_1931(nm);
        
        total[0] += x * reflectance * d65 * step_size;
        total[1] += y * reflectance * d65 * step_size;
        total[2] += z * reflectance * d65 * step_size;
    }
    
    std::cout << "Raw XYZ (before normalization): (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    // Compute normalization
    double normalization = 0.0;
    for (int nm = 380; nm <= 800; nm += step_size) {
        double y = yFit_1931(nm);
        double d65 = getD65Value(nm);
        normalization += y * d65 * step_size;
    }
    
    std::cout << "Normalization constant: " << normalization << std::endl;
    
    // Normalize
    total[0] = 100.0 * total[0] / normalization;
    total[1] = 100.0 * total[1] / normalization;
    total[2] = 100.0 * total[2] / normalization;
    
    std::cout << "Normalized XYZ (should be ~47.5, 50, 54): (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    std::vector<double> rgb = XYZ_to_sRGB(total);
    
    std::cout << "\n✓ Expected: RGB ≈ (188, 188, 188) - Neutral gray" << std::endl;
    std::cout << "✓ Got: RGB = (" << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << ")" << std::endl;
    
    if (std::abs(rgb[0] - 188) < 20 && std::abs(rgb[1] - 188) < 20 && std::abs(rgb[2] - 188) < 20) {
        std::cout << "✓ PASS: Gray color correct!" << std::endl;
    } else {
        std::cout << "✗ FAIL: Gray color incorrect!" << std::endl;
    }
}

void test_typical_skin() {
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          TEST 3: Typical Skin-like Spectrum           ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    int step_size = 5;
    std::vector<double> total = {0.0, 0.0, 0.0};
    
    // Skin-like reflectance: higher in red, lower in blue
    std::cout << "\nSimulating skin-like reflectance spectrum" << std::endl;
    std::cout << "Blue region (380-500nm): ~35% reflectance" << std::endl;
    std::cout << "Green region (500-600nm): ~60% reflectance" << std::endl;
    std::cout << "Red region (600-800nm): ~75% reflectance" << std::endl;
    
    for (int nm = 380; nm <= 800; nm += step_size) {
        double reflectance;
        
        // Skin reflects more red, less blue
        if (nm < 500) {
            reflectance = 0.35;  // Low blue reflectance
        } else if (nm < 600) {
            reflectance = 0.60;  // Medium green reflectance
        } else {
            reflectance = 0.75;  // High red reflectance
        }
        
        double d65 = getD65Value(nm);
        double x = xFit_1931(nm);
        double y = yFit_1931(nm);
        double z = zFit_1931(nm);
        
        total[0] += x * reflectance * d65 * step_size;
        total[1] += y * reflectance * d65 * step_size;
        total[2] += z * reflectance * d65 * step_size;
    }
    
    std::cout << "\nRaw XYZ (before normalization): (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    // Compute normalization
    double normalization = 0.0;
    for (int nm = 380; nm <= 800; nm += step_size) {
        double y = yFit_1931(nm);
        double d65 = getD65Value(nm);
        normalization += y * d65 * step_size;
    }
    
    std::cout << "Normalization constant: " << normalization << std::endl;
    
    // Normalize
    total[0] = 100.0 * total[0] / normalization;
    total[1] = 100.0 * total[1] / normalization;
    total[2] = 100.0 * total[2] / normalization;
    
    std::cout << "Normalized XYZ: (" << total[0] << ", " << total[1] << ", " << total[2] << ")" << std::endl;
    
    std::vector<double> rgb = XYZ_to_sRGB(total);
    
    std::cout << "\n✓ Expected: RGB with R > G > B (peachy/beige skin tone)" << std::endl;
    std::cout << "✓ Got: RGB = (" << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << ")" << std::endl;
    
    if (rgb[0] > rgb[1] && rgb[1] > rgb[2] && rgb[0] > 150) {
        std::cout << "✓ PASS: Skin color has correct R>G>B relationship!" << std::endl;
    } else {
        std::cout << "✗ FAIL: Skin color incorrect! Red should be highest." << std::endl;
    }
}

void test_your_actual_xyz() {
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║      TEST 4: Your Actual XYZ Values from CSV          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    // From your CSV: X=9.91162, Y=7.65895, Z=5.44309
    std::vector<double> xyz = {9.91162, 7.65895, 5.44309};
    
    std::cout << "\nUsing actual XYZ from your CSV file:" << std::endl;
    std::cout << "Input: (" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << ")" << std::endl;
    
    std::vector<double> rgb = XYZ_to_sRGB(xyz);
    
    std::cout << "\n✓ Your CSV shows: RGB = (3.526, 254.757, 249.15)" << std::endl;
    std::cout << "✓ This function produces: RGB = (" << rgb[0] << ", " << rgb[1] << ", " << rgb[2] << ")" << std::endl;
    
    std::cout << "\nANALYSIS:" << std::endl;
    std::cout << "- Your Y value of 7.66 is very low (should be 50-80 for skin)" << std::endl;
    std::cout << "- This suggests normalization is wrong OR reflectance is wrong" << std::endl;
    std::cout << "- The XYZ ratios (X>Y>Z) are correct for skin" << std::endl;
    std::cout << "- But the absolute values are ~10x too small" << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(3);
    
    std::cout << "╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        COLOR CONVERSION DIAGNOSTIC TEST SUITE         ║" << std::endl;
    std::cout << "║                                                        ║" << std::endl;
    std::cout << "║  This will help identify where your color conversion  ║" << std::endl;
    std::cout << "║  pipeline is producing incorrect RGB values.          ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    test_perfect_white();
    test_neutral_gray();
    test_typical_skin();
    test_your_actual_xyz();
    
    std::cout << "\n╔════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                  DIAGNOSTIC COMPLETE                   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════╝" << std::endl;
    
    return 0;
}