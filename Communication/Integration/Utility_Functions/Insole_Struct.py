from ctypes import *

class UserInfo(Structure):
    _fields_ = [
        ('ht', c_uint),
        ('wt', c_uint),
        ('no_of_devices', c_uint),
        ('arch', c_float),
        ('bmi', c_float),
        ('shoe', c_uint),
        ('stressMult', c_float),
        ('isShod', c_int),
        ('reportPressure', c_int),
        ('heel_a', c_float),
        ('heel2_a', c_float),
        ('arch_a', c_float),
        ('mid_a', c_float),
        ('f_a', c_float),
        ('f2_a', c_float),
        ('hallux_a', c_float),
        ('s_toes_a', c_float),
    ]


class RunInfo(Structure):
    _fields_ = [
        ('strideCount', c_uint),
        ('dist', c_float),
        ('time', c_float),
        ('calories', c_uint),
        ('balance', c_int),
        ('currMode', c_int),
        ('peak_total_load', c_float),
        ('currCOB', c_int * 2),
        ('avgCOB', c_int * 2),
    ]


class NewStrideInfo(Structure):
    _fields_ = [
        ('landtime', c_uint),
        ('speed', c_float),
        # Below values are calculated by the library
        ('gct', c_uint),
        ('airtime', c_uint),
        ('strike_type', c_int),
        ('overpronate_deg', c_float),
        ('strideRate', c_uint),
        ('strideLength', c_uint),
        ('count', c_uint),
        ('raw_sc', c_uint),
        ('time', c_uint),
        ('time_dhms', c_uint),
        ('alert', c_int),
        ('data_strength', c_int),
        ('st_begin_t', c_int),
        ('ttfc', c_uint),
        ('hlt', c_uint),
        ('tot', c_uint),
        ('mst', c_uint),
        ('dst_begin', c_uint),
        ('dst_end', c_uint),
        ('dat_begin', c_uint),
        ('dat_end', c_uint),
        ('debug_ble_1', c_ulonglong),
        ('debug_ble_2', c_ulonglong),
        # Below are for internal usage.
        ('impact_a', c_int),
        ('push_a', c_int),
        ('a_x', c_int),
        ('a_y', c_int),
        ('a_z', c_int),
        ('g_x', c_int),
        ('g_y', c_int),
        ('g_z', c_int),
        ('m_x', c_int),
        ('m_y', c_int),
        ('m_z', c_int),
        ('p_p_x', c_int),
        ('p_p_y', c_int),
        ('p_p_z', c_int),
        ('lastStateTime', c_int),
        ('state', c_int),
        ('st_count', c_int * 3),
        ('quat', c_double * 4),
        ('abs_a', c_double * 3),
        ('vel', c_double * 3),
        ('pos', c_double * 3),
    ]


class StrideVariability(Structure):
    _fields_ = [
        ('heel_strike_c', c_uint),
        ('mid_strike_c', c_uint),
        ('front_strike_c', c_uint),
        ('ttfc_avg', c_uint),
        ('ttfc_delta', c_uint),
        ('overpronation_avg', c_float),
        ('hlt_avg', c_uint),
        ('hlt_delta', c_uint),
        ('tot_avg', c_uint),
        ('tot_delta', c_uint),
        ('mst_avg', c_uint),
        ('mst_delta', c_uint),
        ('airtime_avg', c_uint),
        ('airtime_delta', c_uint),
        ('sr_avg', c_uint),
        ('sr_delta', c_uint),
        ('sl_avg', c_uint),
        ('sl_delta', c_uint),
        ('gct_avg', c_uint),
        ('gct_delta', c_uint),
        ('dst_avg', c_uint),
        ('dst_delta', c_uint),
    ]

class StressInfo(Structure):
    _fields_ = [
        ('heel', c_float),
        ('heel2', c_float),
        ('mid', c_float),
        ('arch', c_float),
        ('plantar', c_float),
        ('front', c_float),
        ('front_2', c_float),
        ('hallux', c_float),
        ('toes', c_float),
        ('knee', c_float),
        ('knee_s', c_float),
        ('total', c_float),
        # Below store push-off forces
        ('pushoff', c_float),
        # Below contain Peak impact value since last micro-reset
        ('peak_front', c_float),
        ('peak_front_2', c_float),
        ('peak_hallux', c_float),
        ('peak_toes', c_float),
        ('peak_mid', c_float),
        ('peak_heel', c_float),
        ('peak_heel2', c_float),
        ('peak_arch', c_float),
        ('peak_knee', c_float),
        ('peak_total', c_float),
        ('peak_knee_s', c_float),
        # Below store loading rate (in KG/s)
        ('lr_heel', c_float),
        ('lr_mid', c_float),
        ('lr_front', c_float),
        ('lr_total', c_float),
        # Below store power (in Watts)
        ('power_total', c_float),
        ('peak_power_total', c_float),
    ]

class MatrixLoadInfo(Structure):
    _fields_ = [
        ('load_inst', (c_int * 9) * 20),
        ('peak_load', (c_int * 9) * 20),
        ('cob', c_int * 2),
    ]
