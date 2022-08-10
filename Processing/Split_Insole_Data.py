##
import pandas as pd

##
data_file_name = 'subject0_20220805_175751'

static_transition_data = pd.DataFrame({
    'locomotion_mode': ["SSLW", "LWSS", "SASS", "SSSA", "SDSS", "SSSD"],
    'locomotion_start': [60000,],
    'locomotion_end': [61300,]
})

dynamic_transition_data = pd.DataFrame({
    'locomotion_mode': ["LWSA", "SALW", "LWSD", "SDLW"],
    'leading_leg': [""],
    'locomotion_start': [64030, ],
    'locomotion_end': [64780, ]
})

steady_state_data = pd.DataFrame({
    'locomotion_mode': ["SS", "LW", "SA", "SD"],
    'locomotion_start': [55000, 61300, ],
    'locomotion_end': [60000, 64030,  ]
})


split_data = pd.DataFrame({
    'locomotion_mode': ["SS", "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW",
                        "SSLW", "LW", "LWSA", "SA", "SASS", "turnSASD", "SSSD", "SD", "SDLW", "LW", "LWSS", "turnLWLW"],
    'locomotion_start': [55000, 60000, 61300, 64030, 64800, 70000, 70900, 78400, ],
    'locomotion_end': [60000, 61300, 64030, 64800, 70000, 70900, 78400]
})

split_data = pd.DataFrame({
    'SS': [55000],
    'SSLW': [60000, ],
    'LW': [61300, ],
    'LWSA': [64030, ],
    'SA': [64800, ],
    'SASS': [70000, ],
    'turnSASD': [70900, ],
    'SSSD': [78400, ],
    'SD': [],
    'SDLW': [],
    'LW': [],
    'LWSS': []
})
