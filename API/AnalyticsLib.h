

#ifdef __cplusplus
extern "C" {
#endif
    //
    //  AnalyticsLib.h
    //  Stridalyzer
    //
    //  Created by Anshuman - ReTiSense on 11/01/15.
    //  Copyright (c) 2015 Stridalyzer. All rights reserved.
    //
    /*
     #include <stdlib.h>
     #include <string.h>
     #include <string.h>
     */

#ifndef Stridalyzer_AnalyticsLib_h
#define Stridalyzer_AnalyticsLib_h

// Top-level header file for Analytics Library


struct UserInfo {
    unsigned int ht;    // Height in cm
    unsigned int wt;    // Weight in grams
    unsigned int no_of_devices;
    float arch;  // Arch height in mm
    float bmi;
    unsigned int shoe;  // Shoe size in cm
    float stressMult;
    int isShod; // whether the user is wearing sports shoes
    int reportPressure;  // Pressure is reported in kPa
    float heel_a;
    float heel2_a;
    float arch_a;
    float mid_a;
    float f_a;
    float f2_a;
    float hallux_a;
    float s_toes_a;
};

struct RunInfo {
    unsigned int strideCount;
    float dist; // In meters
    float time; // in sec.s
    unsigned int calories;
    int balance; // varies from -100 to +100 (percent of weight)
    int currMode;
    float peak_total_load;
    // Center-of-balance for the analysis, accounting for both feet.
    // [0] represents horizontal position, and [1] represents vertical position (both relative)
    // each value goes from -100 to +100
    // for horizontal: -100 (outer edge of left foot) to 0 (inner edge of both feet) to +100 (outer edge of right foot) - no distance is assumed between feet.
    // For vertical: -100 (center of heel) to +100 (center of toes)
    int currCOB[2];
    int avgCOB[2];
};

enum StrikeType {
    HEEL,
    MIDFOOT,
    FRONT
};

enum AlertType {
    NONE,
    OVERSTRIDE,
    OVERPRONATE,
    KNEE_STRESS,
    ARCH_STRESS,
    FRONT_STRESS,
    HEEL_STRESS,
    MID_STRESS,
    STRIKE_CHANGE
};

//extern unsigned int stride_process_count; // Number of strides to accumulate before processing
extern unsigned int debugMode;

    struct NewStrideInfo {
        // Landing time is time "0" for a stride.
        unsigned int landtime;
        float speed; // in m/s
        // Below values are calculated by the library
        unsigned int gct;           // Ground contact Time of the current stride (in ms)
        unsigned int airtime;       // (trailing) airtime of the current stride (in ms)
        enum StrikeType strike_type;// Stores the Strike type of the current stride.
        float overpronate_deg;        // 1 if overpronated.
        unsigned int strideRate;    // Cadence (in steps per min)
        unsigned int strideLength;  // Stride Length (in cm)
        unsigned int count;         // Count of strides of this side (left/right)
        unsigned int raw_sc;        // stepcount obtained from insole 
        unsigned int time;                   // on-board time (in ms) received from insole, from the latest dataset processed.
        unsigned int time_dhms;              // (where supported) Time in day/hour/min/second format. 0 if not supported
        enum AlertType alert;
        int data_strength;
        int st_begin_t;
        unsigned int ttfc;          // Time to full contact (ms) [time from landing to full-contact]
        unsigned int hlt;           // Heel lift time (ms) [time from landing to heel-lift]
        unsigned int tot;           // Toe-off time (ms) [time from landing to toe-off]
        unsigned int mst;           // mid-swing time (ms) 
        unsigned int dst_begin;     // (leading) start of double-support BY current foot.
        unsigned int dst_end;       // (leading) end of double-support BY other foot.
        unsigned int dat_begin;     // (leading) start of double-airtime BY current foot.
        unsigned int dat_end;       // (leading) end of double-air BY other foot.
        unsigned long long debug_ble_1;
        unsigned long long debug_ble_2;

        // Below are for internal usage.
        int impact_a;
        int push_a;
        int a_x;
        int a_y;
        int a_z;
        int g_x;
        int g_y;
        int g_z;
        int m_x;
        int m_y;
        int m_z;
        int p_p_x;
        int p_p_y;
        int p_p_z;
        int lastStateTime;
        int state;
        int st_count[3];
        double quat[4];
        double abs_a[3];
        double vel[3];
        double pos[3];
    };

    struct StrideVariability {
        unsigned int heel_strike_c;
        unsigned int mid_strike_c;
        unsigned int front_strike_c;
        unsigned int ttfc_avg;
        unsigned int ttfc_delta;
        float overpronation_avg;
        unsigned int hlt_avg;
        unsigned int hlt_delta;
        unsigned int tot_avg;
        unsigned int tot_delta;
        unsigned int mst_avg;
        unsigned int mst_delta;
        unsigned int airtime_avg;
        unsigned int airtime_delta;
        unsigned int sr_avg;
        unsigned int sr_delta;
        unsigned int sl_avg;
        unsigned int sl_delta;
        unsigned int gct_avg;
        unsigned int gct_delta;
        unsigned int dst_avg;
        unsigned int dst_delta;
    };

struct StressInfo {
    // Below contain force/pressure values. Force values will be in KG, pressure values in kPa
    float heel;
    float heel2;
    float mid;          // Mid is Outer midfoot.
    float arch;         // Arch is inner midfoot
    float plantar;      // Plantar pressure. Typically equal to arch.
    float front;        // Metatarsals 1,2
    float front_2;      // Metatarsals 3-5
    float hallux;
    float toes;
    float knee;         // Vertical forces at knee
    float knee_s;       // Lateral forces at knee
    float total;        // Total ground reaction force.
    // Below store push-off forces
    float pushoff;
    // Below contain Peak impact value since last micro-reset
    float peak_front;
    float peak_front_2;
    float peak_hallux;
    float peak_toes;
    float peak_mid;
    float peak_heel;
    float peak_heel2;
    float peak_arch;
    float peak_knee;
    float peak_total;
    float peak_knee_s;
    // Below store loading rate (in KG/s)
    float lr_heel;
    float lr_mid;
    float lr_front;
    float lr_total;
    // Below store power (in Watts)
    float power_total;
    float peak_power_total;
};
 
#define M_ROWS 20
#define M_COLS 9
    struct MatrixLoadInfo {
        int load_inst[M_ROWS][M_COLS];   // instantaneous Load values in 1 gm (0.001KG) units. 20 rows & 9 cols
        int peak_load[M_ROWS][M_COLS];   // Peak Load values
        int cob[2]; // Center-of-balance location on the grid. [0] represents horizontal position, and [1] represents vertical position

    };
    

extern const unsigned int IMPACT_MAX;
extern const float Pushoff_Ratio; // 50% of GCT
extern const float Pushoff_Ratio_Walk; // 30% of GCT
extern const unsigned int MIN_SPEED;
extern const unsigned int MAX_WALK_SPEED;
extern const unsigned int MAX_SPEED;
extern const float MIN_MULT;
extern const unsigned int MAX_MULT;
extern const unsigned int MAX_INTERVAL;
extern const float HEEL_TOE_TIME;
extern const int _CONSTRAIN;

// Height in cm, wt in gm, Arch height in mm, shoe size in cm, isShod is for whether the user is wearing sports shoes
void InitUser (int ht, int wt, int arch, int footsize, int no_of_devices, int isShod, int reportPressure);
    
void ResetCalibration ();
void SetCalibration (int method, int phase, float wt); // wt in KG
int isInsoleCalibrated (int isLeft);
// Get calibration ratios (encrypted). If there is valid calibration data, return value will be 1, and the pointers will be set.
// It is assumed that l_cal and r_cal point to valid long ints
int getCalibrationRatios (unsigned long* l_cal_low, unsigned long* r_cal_low, unsigned long* l_cal_high, unsigned long* r_cal_high);
int setCalibrationRatios (unsigned long l_cal_low, unsigned long r_cal_low, unsigned long l_cal_high, unsigned long r_cal_high);

// l_mult and r_mult are multipliers to the load value, to linearly scale the actual sensor value to get the correct sensor value.
// l_thresh and r_thresh are threshold values (in grams) of valid load. So, any load value below threshold will be set to 0.
// For PRISM, this is applied individually to all sensors.
// Get calibration ratios. If there is valid calibration data, return value will be 1, and the pointers will be set.
int getCalibrationRatiosSimple (float* l_mult, int* l_thresh, float* r_mult, int* r_thresh);
// Set calibration values : set the calibration values for analytics.
int setCalibrationRatiosSimple (float l_mult, int l_thresh, float r_mult, int r_thresh);

void SetSensorSpec (int f1, int f2, int m, int a, int h, int t, int f1_type, int f2_type, int m_type, int a_type, int h_type, int t_type, int rev, int isLeft);
void SetSensorSpecNew (int f1, int f2, int m, int a, int h, int h2, int t, int st, int sensorType, int rev, int isLeft);
void setLinearMult (float l_mult, float r_mult);
struct UserInfo* GetUserInfo (void);
struct RunInfo* GetRunInfo (void);
float CurrStrideLen (void);  // in meters
float CurrStrideTime (int isMs); // In sec.c


struct NewStrideInfo* GetNewStrideInfo (unsigned int isLeft);
struct NewStrideInfo* GetLastFullStride (unsigned int isLeft);
struct StrideVariability* GetStrideVariabilityInfo (unsigned int isLeft);

//struct StrideInfo* GetStrideInfo (unsigned int isLeft);
struct StressInfo* GetStressInfo (unsigned int isLeft);
struct MatrixLoadInfo* GetMatrixLoadInfo (unsigned int isLeft);
struct MatrixLoadInfo* GetAvgMatrixLoadInfo (unsigned int isLeft);

struct StrideInfo* GetAvgStride (unsigned int isLeft);
struct NewStrideInfo* GetAvgNewStride (unsigned int isLeft);
struct StressInfo* GetAvgStress (unsigned int isLeft);
void   ResetStrideInfo (int isShod);
void   ResetIMUData (int isLeft, int resetType);  // resetType : 1 for hard reset, 2 for ZUPT (ground contact) reset
void   UpdateLastStrideInfo (void);


// Process discrete sensor data. Returns 0 if something is wrong, or if no new step is detected (dynamic mode).
// Returns stepcount (dynamic mode) for every new step, and 1 for all successful processing (static mode)
int ProcessStride_FSR (unsigned char* bytes, int l, float speed_m_s, float d_m, float alt_m, int isLeft, int mode, int isReset, int isInsight, int noConstrain);
int ProcessStride_FSR_step (unsigned char* bytes, int l, float speed_m_s, float d_m, float alt_m, int isLeft, int mode, int isReset, int isInsight, int noConstrain);
int ProcessStride_acc_gyro (unsigned char* bytes, int l, int isLeft, int mode);
int ProcessStride_FSR_grid (unsigned char* bytes, int l, float speed_m_s, float d_m, float alt_m, int isLeft, int mode, int isReset, int noConstrain);
int ProcessStride_FSR_bb_grid (unsigned char* bytes, int l, int isLeft, int mode, int isReset, int noConstrain);

int ProcessBLEData (unsigned char* bytes, int l, int isLeft, int mode);


    
float constrain_value (float v_last, float v_curr, float band, int allowIncrease);

void SetDebugMode (unsigned int mode);
int GetRevNumber (void);
void test_analyticslib (void);
    
// Other helper functions to ease data access

unsigned int GetRunInfo_strideCount();
// Get the COB indices (one at a time). isCurr: (0: avg, 1: current) isLeft : self-explanatory isRow : (0: return column index 1: return row index)
int GetRunInfo_COB (int isCurr, int isRow);
// current Stride accessors
unsigned int GetStride_gct(int isCurr, int isLeft);           // Ground contact Time of the current stride (in ms)
unsigned int GetStride_airtime(int isCurr, int isLeft);       // (trailing) airtime of the current stride (in ms)
float GetStride_overpronate_deg(int isCurr, int isLeft);        // 1 if overpronated.
unsigned int GetStride_strikeCount(int isLeft, int index);
unsigned int GetStride_strideRate(int isCurr, int isLeft);    // Cadence (in steps per min)
unsigned int GetStride_strideLength(int isCurr, int isLeft);  // Stride Length (in cm)
unsigned int GetStride_ttfc(int isCurr, int isLeft);
unsigned int GetStride_hlt(int isCurr, int isLeft);
unsigned int GetStride_tot(int isCurr, int isLeft);
unsigned int GetStride_mst(int isCurr, int isLeft);
unsigned int GetStride_dst(int isCurr, int isLeft);
unsigned int GetStride_dat(int isCurr, int isLeft);
unsigned int GetStride_state(int isCurr, int isLeft);


double GetIMUQuat (int isLeft, int quatIndex);
double GetIMUAccel (int isLeft, int a_i);
double GetIMUVel (int isLeft, int v_i);
double GetIMUPos (int isLeft, int p_i);
void MapAxis (int real_x, int real_y, int real_z, int dir_x, int dir_y, int dir_z);

unsigned int GetCurrStride_strike_type(int isLeft);             // Stores the Strike type of the current stride.
unsigned int GetCurrStride_count(int isLeft);         // Count of strides of this side (left/right)
int GetCurrStride_time(int isLeft);                   // on-board time (in ms) received from insole, from the latest dataset processed.
int GetCurrStride_a_x(int isLeft);
int GetCurrStride_a_y(int isLeft);
int GetCurrStride_a_z(int isLeft);
int GetCurrStride_g_x(int isLeft);
int GetCurrStride_g_y(int isLeft);
int GetCurrStride_g_z(int isLeft);

// current Stress accessors
float GetStress_heel(int isCurr, int isLeft);
float GetStress_heel2(int isCurr, int isLeft);
float GetStress_mid(int isCurr, int isLeft);          // Mid is Outer midfoot.
float GetStress_arch(int isCurr, int isLeft);         // Arch is inner midfoot
float GetStress_plantar(int isCurr, int isLeft);      // Plantar pressure. Typically equal to arch.
float GetStress_front(int isCurr, int isLeft);        // Metatarsals 1,2
float GetStress_front_2(int isCurr, int isLeft);      // Metatarsals 3-5
float GetStress_hallux(int isCurr, int isLeft);
float GetStress_toes(int isCurr, int isLeft);
float GetStress_total(int isCurr, int isLeft);        // Total ground reaction force.
// Below contain Peak impact value since last micro-reset
float GetCurrStress_peak_front(int isLeft);
float GetCurrStress_peak_front_2(int isLeft);
float GetCurrStress_peak_hallux(int isLeft);
float GetCurrStress_peak_toes(int isLeft);
float GetCurrStress_peak_mid(int isLeft);
float GetCurrStress_peak_heel(int isLeft);
float GetCurrStress_peak_heel2(int isLeft);
float GetCurrStress_peak_arch(int isLeft);

// Direct accessors for MatrixLoadInfo
int* GetMatrixLoadInfo_array (int isCurr, int isPeak, int isLeft);
int  GetMatrixLoadInfo_cell (int isCurr, int isPeak, int isLeft, int row, int col);
int  GetMatrixLoadInfo_halfcell (int isCurr, int isPeak, int isLeft, int row, int col);
// Get the COB indices (one at a time). isCurr: (0: avg, 1: current) isLeft : self-explanatory isRow : (0: return column index 1: return row index)
int  GetMatrixLoadInfo_COB (int isCurr, int isLeft, int isRow);

#endif


#ifdef __cplusplus
};
#endif
