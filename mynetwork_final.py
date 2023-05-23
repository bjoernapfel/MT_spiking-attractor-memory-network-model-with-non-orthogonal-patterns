import math
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import random
from time import perf_counter
from pygenn import genn_model

# How many HC / MC
NUM_HC = 10 #scaled to 10, 15, 20 for Fig. 6
NUM_MC = 10

#Simulation time step
DT = 0.1 #ms

MEASURE_TIMING = True

### Simulating input
#encoding phase
activation_duration = 300 #ms
delay_duration = 400 #ms
encoding_duration = (activation_duration + delay_duration) * NUM_MC
#free recall phase
free_recall_duration = encoding_duration #ms
cue_duration = 20 #ms
disruptor_duration = 300 #ms
#Stiulation duration
duration_time = (activation_duration + delay_duration) * NUM_MC + free_recall_duration
duration_time_sec = duration_time/1000

#overlap
overlap_percentage = 4 #scaled from 0-10 for Fig. 3-5; at 4 for Fig. 6 can not be 0.0 for controlled overlap

### Neurons for neuron population
#aif parameters
PN_alif_PARAMS = {"tauM": 20.0,
                    "R_m": 220.0,
                    "C_m": 280,
                    "tau_w": 500.0,
                    "V_thresh": -55.0,
                    "V_rest": -60.0,
                    "tau_Ref": 5.0,
                    "a": 0.0,
                    "b": 86.0}

DBC_alif_PARAMS = {"tauM": 20.0,
                    "R_m": 660.0,
                    "C_m": 15.0,
                    "tau_w": 200.0,
                    "V_thresh": -44.0,
                    "V_rest": -60.0,
                    "tau_Ref": 2.0,
                    "a": 0.0,
                    "b": 3.0}

BC_alif_PARAMS = copy(DBC_alif_PARAMS)
BC_alif_PARAMS["b"]: 0.0

PN_alif_init = {"V": PN_alif_PARAMS["V_rest"], "RefracTime": 0.0, "w": 0.0}
DBC_alif_init = {"V": DBC_alif_PARAMS["V_rest"], "RefracTime": 0.0, "w": 0.0}
BC_alif_init = {"V": BC_alif_PARAMS["V_rest"], "RefracTime": 0.0, "w": 0.0}

#alif neuron model
alif = genn_model.create_custom_neuron_class(
    "alif",
    param_names = ["tauM", "R_m", "C_m", "tau_w", "V_thresh", "V_rest", "tau_Ref", "a", "b"],
    var_name_types = [("V", "scalar"), ("RefracTime", "scalar"), ("w", "scalar")],
    derived_params= [["ExpTC", genn_model.create_dpf_class(lambda pars, dt : np.exp(-dt / pars[0]))()],
                    ["ExpTC_w", genn_model.create_dpf_class(lambda pars, dt : np.exp(-dt / pars[3]))()],
                    ["Rmembrane", genn_model.create_dpf_class(lambda pars, dt : pars[0] / pars[2])()]],
    sim_code = 
    """
    if ($(RefracTime) <= 0.0) {
        scalar alpha = ($(Isyn) * $(Rmembrane)) + $(V_rest) - ($(w) * $(Rmembrane));
        $(V) = alpha - ($(ExpTC) * (alpha - $(V)));
        scalar beta = $(a) * ($(V) - $(V_rest)) - $(w);
        $(w) = beta - ($(ExpTC_w) * (beta - $(w)));
    } else {
        $(RefracTime) -= DT;
    }
    """,
    threshold_condition_code =
    """
    $(RefracTime) <= 0.0 && $(V) >= $(V_thresh) 
    """,
    reset_code = 
    """
    $(V) = $(V_rest);
    $(w) += $(b);
    $(RefracTime) = $(tau_Ref);
    """)

### Synapses for synapse population / update model
#BCPNN parameters
BCPNN_PARAMS = {"tauZi": 5.0, #ms
                "tauZj": 5.0, #ms
                "tauP": 5000.0, #ms
                "fMax": 25.0,
                "encoding_duration": (activation_duration + delay_duration) * NUM_MC,
                "wMin": 0.0,
                "wMax": 400.0,
                "wGain": 1.0} #nS

BCPNN_PARAMS_bh = {"tauZi": 5.0, #ms
                    "tauZj": 5.0, #ms
                    "tauP": 5000.0, #ms
                    "fMax": 25.0,
                    "encoding_duration": (activation_duration + delay_duration) * NUM_MC,
                    "wMin": 0.0,
                    "wMax": 400.0,
                    "wGain": 2.5} #nS

BCPNN = genn_model.create_custom_weight_update_class(
    "BCPNN",
    param_names = ["tauZi", "tauZj", "tauP", "fMax", "encoding_duration", "wGain", "wMax", "wMin"],
    var_name_types = [("g", "scalar"), ("PijStar", "scalar")],
    pre_var_name_types = [("ZiStar", "scalar"), ("PiStar", "scalar")],
    post_var_name_types = [("ZjStar", "scalar"), ("PjStar", "scalar")],
    derived_params=[["Ai", genn_model.create_dpf_class(lambda pars, dt: pars[2] / (pars[3] * (pars[0] - pars[2])))()],
                    ["Aj", genn_model.create_dpf_class(lambda pars, dt: pars[2] / (pars[3] * (pars[1] - pars[2])))()],
                    ["Aij", genn_model.create_dpf_class(lambda pars, dt: (pars[2]**2 / (pars[0] + pars[1])) / ((pars[3] * pars[3]) * ((1.0 / ((1.0 / pars[0]) + (1.0 / pars[1]))) - pars[2])))()],
                    ["Epsilon", genn_model.create_dpf_class(lambda pars, dt: pars[2] / (pars[3] * pars[2]))()]], 
    sim_code =
    """
    if ($(t) >= $(encoding_duration)) {
        $(addToInSyn, $(g));
    }
    if ($(t) <= $(encoding_duration)) {
        const scalar timeSinceLastUpdate = $(t) - fmax($(prev_sT_pre), $(sT_post));
        const scalar timeSinceLastPost = $(t) - $(sT_post);
        const scalar newZjStar = $(ZjStar) * exp(-timeSinceLastPost / $(tauZj));
        const scalar newPjStar = $(PjStar) * exp(-timeSinceLastPost / $(tauP));
        $(PijStar) = ($(PijStar) * exp(-timeSinceLastUpdate / $(tauP))) + newZjStar;
        const scalar Pi = $(Ai) * ($(ZiStar) - $(PiStar));
        const scalar Pj = $(Aj) * (newZjStar - newPjStar);
        const scalar Pij = $(Aij) * (($(ZiStar) * newZjStar) - $(PijStar));
        const scalar logPij = log(Pij + ($(Epsilon) * $(Epsilon)));
        const scalar logPiPj = log((Pi + $(Epsilon)) * (Pj + $(Epsilon)));
        const scalar newWeight = $(wGain) * (logPij - logPiPj);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight));
    }
    """,
    learn_post_code=
    """
    if ($(t) < $(encoding_duration)) {
        const scalar timeSinceLastUpdate = $(t) - fmax($(sT_pre), $(prev_sT_post));
        const scalar timeSinceLastPre = $(t) - $(sT_pre); 
        const scalar newZiStar = $(ZiStar) * exp(-timeSinceLastPre / $(tauZi));
        const scalar newPiStar = $(PiStar) * exp(-timeSinceLastPre / $(tauP));
        $(PijStar) = ($(PijStar) * exp(-timeSinceLastUpdate / $(tauP))) + newZiStar;
        const scalar Pi = $(Ai) * (newZiStar - newPiStar);
        const scalar Pj = $(Aj) * ($(ZjStar) - $(PjStar));
        const scalar Pij = $(Aij) * ((newZiStar * $(ZjStar)) - $(PijStar));
        const scalar logPij = log(Pij + ($(Epsilon) * $(Epsilon)));
        const scalar logPiPj = log((Pi + $(Epsilon)) * (Pj + $(Epsilon)));
        const scalar newWeight = $(wGain) * (logPij - logPiPj);
        $(g) = fmin($(wMax), fmax($(wMin), newWeight)); //fmin(max): what ever value is smaller(bigger)is taken
    }
    """,
    pre_spike_code =
    """
    if ($(t) < $(encoding_duration)) {
        const scalar dt = $(t) - $(sT_pre);
        $(ZiStar) = ($(ZiStar) * exp(-dt / $(tauZi))) + 1.0;
        $(PiStar) = ($(PiStar) * exp(-dt / $(tauP))) + 1.0;
    }
    """,
    post_spike_code = 
    """
    if ($(t) < $(encoding_duration)) {
        const scalar dt = $(t) - $(sT_post);
        $(ZjStar) = ($(ZjStar) * exp(-dt / $(tauZj))) + 1.0;
        $(PjStar) = ($(PjStar) * exp(-dt / $(tauP))) + 1.0;
    }
    """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True,
    is_prev_pre_spike_time_required=True,
    is_prev_post_spike_time_required=True)

#STDP_paremeters
stdp_additive_AMPA_PARAMS = {"tauPlus": 20.0,
                            "tauMinus": 20.0,
                            "aPlus": 0.1, 
                            "aMinus": 0.15,
                            "alpha": 1, 
                            "wMin": 0.0,
                            "wMax": 1000}
stdp_additive_NMDA_PARAMS = {"tauPlus": 20.0,
                            "tauMinus": 20.0,
                            "aPlus": 0.1,
                            "aMinus": 0.15,
                            "alpha": 1, 
                            "wMin": 0.0,
                            "wMax": 1000}

#STDP weight update rule
stdp_additive = genn_model.create_custom_weight_update_class(
    "stdp_additive",
    param_names=["tauPlus", "tauMinus", "aPlus", "aMinus", "alpha", "wMin", "wMax"],
    var_name_types=[("g", "scalar")],
    sim_code="""
        $(addToInSyn, $(g));
        const scalar dt = $(t) - $(sT_post);
        if (dt > 0) {
            const scalar timing = exp(-dt / $(tauMinus));
            const scalar newWeight = $(g) * $(alpha) - ($(aMinus) * timing);
            $(g) = fmax($(wMin), fmin($(wMax), newWeight));
        }
        """,
    learn_post_code="""
        const scalar dt = $(t) - $(sT_pre);
        if (dt > 0) {
            const scalar timing = exp(-dt / $(tauPlus));
            const scalar newWeight = $(g) + ($(aPlus) * timing);
            $(g) = fmax($(wMin), fmin($(wMax), newWeight));
        }
        """,
    is_pre_spike_time_required=True,
    is_post_spike_time_required=True)
 
# Current source model
PoissonExp_STIM = genn_model.create_custom_current_source_class(
    "PoissonExp_STIM",
    param_names=["weight", "tauSyn", "rate", "STIM_start", "STIM_stop"],
    var_name_types=[("current", "scalar")],
    derived_params=[["ExpDecay", genn_model.create_dpf_class(lambda pars, dt : math.exp(-dt / pars[1]))()],
                    ["Init", genn_model.create_dpf_class(lambda pars, dt : pars[0] * (1.0 - math.exp(-dt / pars[1])) * (pars[1] / dt))()],
                    ["ExpMinusLambda", genn_model.create_dpf_class(lambda pars, dt : math.exp(-(pars[2] / 1000.0) * dt))()]],
    injection_code="""
    if ($(t) >= $(STIM_start) && $(t) <= $(STIM_stop)) {
        scalar p = 1;
        unsigned int numSpikes = 0;
        while (p > $(ExpMinusLambda)) {
            numSpikes += 1;
            p *= $(gennrand_uniform);
            };
        $(current) += $(Init) * (numSpikes - 1);
        $(injectCurrent, $(current));
        $(current) *= $(ExpDecay);
    }
    """) 
    
# Current source model used exclusively for delayed-response task with distractor in Fig. 4, 5B and 5D
PoissonExp_STIM_disruptor = genn_model.create_custom_current_source_class(
    "PoissonExp_STIM",
    param_names=["weight", "tauSyn", "rate", "STIM_start", "STIM_stop"],
    var_name_types=[("current", "scalar")],
    derived_params=[["ExpDecay", genn_model.create_dpf_class(lambda pars, dt : math.exp(-dt / pars[1]))()],
                    ["Init", genn_model.create_dpf_class(lambda pars, dt : pars[0] * (1.0 - math.exp(-dt / pars[1])) * (pars[1] / dt))()],
                    ["ExpMinusLambda", genn_model.create_dpf_class(lambda pars, dt : math.exp(-(pars[2] / 1000.0) * dt))()]],
    injection_code="""
    if ($(t) >= $(STIM_start) && $(t) <= $(STIM_stop) && $(gennrand_uniform) > 0.15) {
        scalar p = 1;
        unsigned int numSpikes = 0;
        while (p > $(ExpMinusLambda)) {
            numSpikes += 1;
            p *= $(gennrand_uniform);
            };
        $(current) += $(Init) * (numSpikes - 1);
        $(injectCurrent, $(current));
        $(current) *= $(ExpDecay);
    }
    """)
        
# Create model
model = genn_model.GeNNModel("float", "Hypercolum")
model.dT = DT
model.timing_enabled = MEASURE_TIMING

# PARAMETER NEURON POP
# number of neurons per population
NUM_PN = 30
NUM_DBC = 1
NUM_BC = 4

# initial values neuron variables
PN_init = {"V": PN_alif_PARAMS["V_rest"], "RefracTime": 0.0}
DBC_init = {"V": DBC_alif_PARAMS["V_rest"], "RefracTime": 0.0}
BC_init = copy(DBC_init)

# neuron pop names over HCs
PN_MC__HC = []
DBC_MC__HC = []
BC_HC = []
z=0

# PARAMETER SYNAPSE POP
# weight update rule
e2i_PN_BC = "StaticPulse"
i2e_BC_PN = "StaticPulse"
i2e_DBC_PN = "StaticPulse"
zmn_synapse = "StaticPulse"

# initial values update rule variables
f_desired=1
f_max=25
BCPNN_var_init = {"g": 0.0,
                  "PijStar": 0.0} 
BCPNN_pre_var_init = {"ZiStar": 0.0,
                      "PiStar": 0.0} 
BCPNN_post_var_init = {"ZjStar": 0.0,
                       "PjStar": 0.0}
stdp_additive_var_init = {"g": 0}

# connection probablility between neurons
prob_PN_PN = {"prob": 0.2}
prob_PN_DBC = {"prob": 0.2}
prob_PN_BC = {"prob": 0.7}
prob_PN_PN_bc = {"prob": 0.2}

# decay in post neuron
AMPA_decay = 5.0 #ms
GABA_decay = 10.0 #ms
NMDA_decay = 100.0 #ms

# synaptic delay
delay_within_HC = int(1.5/DT)
delay_between_HC = int(4.5/DT)

# synapse pop names over HCs
PN_MC_PN_MC__HC_AMPA = []
PN_MC_PN_MC__HC_NMDA = []
DBC_MC_PN_MC__HC = []
PN_MCx_DBC_MCy_HC_AMPA = []
PN_MCy_DBC_MCx_HC_AMPA = []
PN_MCx_DBC_MCy_HC_NMDA = []
PN_MCy_DBC_MCx_HC_NMDA = []
s = 0
PN_MC_BC__HC_AMPA = []
PN_MC_BC__HC_NMDA = []
BC_PN_MC__HC = []

PNx_PNy_between_HCs_AMPA = []
PNy_PNx_between_HCs_AMPA = []
PNx_PNy_between_HCs_NMDA = []
PNy_PNx_between_HCs_NMDA = []
PNx_DBCy_between_HCs_a_AMPA = []
PNy_DBCx_between_HCs_a_AMPA = []
PNx_DBCy_between_HCs_a_NMDA = []
PNy_DBCx_between_HCs_a_NMDA = []
PNx_DBCy_between_HCs_b_AMPA = []
PNy_DBCx_between_HCs_b_AMPA = []
PNx_DBCy_between_HCs_b_NMDA = []
PNy_DBCx_between_HCs_b_NMDA = []
m = 0
n = 0
o = 0
all_MC_averages=[]
all_MC_nz_averages=[]

# PARAMETERS ZERO MEAN NOISE

#without distractor
ZMNe_PN_PARAMS = {"weight": 1300, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNi_PN_PARAMS = {"weight": -1300, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNe_DBC_PARAMS = {"weight": 200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNi_DBC_PARAMS = {"weight": -200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}

# zmn neuron / synapse pop names over HCs
ZMNe_PN_HC = []
ZMNi_PN_HC = []
ZMNe_DBC_HC = []
ZMNi_DBC_HC = []

#with distractor
ZMNe_PN_PARAMS_bef_dis = {"weight": 1300, #nS
                 "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": encoding_duration/DT}
ZMNi_PN_PARAMS_bef_dis = {"weight": -1300, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": encoding_duration/DT}
ZMNe_DBC_PARAMS_bef_dis = {"weight": 200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": encoding_duration/DT}
ZMNi_DBC_PARAMS_bef_dis = {"weight": -200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": encoding_duration/DT}

ZMN_potentiation = 10
ZMNe_PN_PARAMS_dis = {"weight": 1300 * ZMN_potentiation, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration,
                  "STIM_stop": encoding_duration + disruptor_duration}
ZMNi_PN_PARAMS_dis = {"weight": -1300 * ZMN_potentiation, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration,
                  "STIM_stop": encoding_duration + disruptor_duration}
ZMNe_DBC_PARAMS_dis = {"weight": 200 * ZMN_potentiation, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration,
                  "STIM_stop": encoding_duration + disruptor_duration}
ZMNi_DBC_PARAMS_dis = {"weight": -200 * ZMN_potentiation, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration,
                  "STIM_stop": encoding_duration + disruptor_duration}

ZMNe_PN_PARAMS_aft_dis = {"weight": 1300, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration + disruptor_duration,
                  "STIM_stop": duration_time}
ZMNi_PN_PARAMS_aft_dis = {"weight": -1300, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration + disruptor_duration,
                  "STIM_stop": duration_time}
ZMNe_DBC_PARAMS_aft_dis = {"weight": 200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration + disruptor_duration,
                  "STIM_stop": duration_time}
ZMNi_DBC_PARAMS_aft_dis = {"weight": -200, #nS
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": encoding_duration + disruptor_duration,
                  "STIM_stop": duration_time}

ZMNe_PN_HC_bef_dis = []
ZMNi_PN_HC_bef_dis = []
ZMNe_DBC_HC_bef_dis = []
ZMNi_DBC_HC_bef_dis = []

ZMNe_PN_HC_dis = []
ZMNi_PN_HC_dis = []
ZMNe_DBC_HC_dis = []
ZMNi_DBC_HC_dis = []

ZMNe_PN_HC_aft_dis = []
ZMNi_PN_HC_aft_dis = []
ZMNe_DBC_HC_aft_dis = []
ZMNi_DBC_HC_aft_dis = []


# PARAMETERS CURRENT SOURCE
PARAMS_STIM_PN = []
PARAMS_STIM_DBC = []

for u in range (NUM_MC):
    weight_potentiation = 3000#200
    PARAMS_STIM_PN.append("PARAMS_PN_STIM" + str(u))
    PARAMS_STIM_DBC.append("PARAMS_DBC_STIM" + str(u))
    PARAMS_STIM_PN[u] = {"weight": 1.5 * weight_potentiation, #nS
                         "tauSyn": 0.1, #ms
                         "rate": 1700,
                         "STIM_start": 0.0 + u*(activation_duration + delay_duration),
                         "STIM_stop": activation_duration + u*(activation_duration + delay_duration)}
    PARAMS_STIM_DBC[u] = {"weight": 0.8 * weight_potentiation, #nS
                          "tauSyn": 0.1, #ms
                          "rate": 75,
                         "STIM_start": 0.0 + u*(activation_duration + delay_duration),
                         "STIM_stop": activation_duration + u*(activation_duration + delay_duration)}
    
# overlap current
PARAMS_STIM_PN_overlap = []
PARAMS_STIM_DBC_overlap = []

for u in range (NUM_MC):
    weight_potentiation = 1.0
    PARAMS_STIM_PN_overlap.append("PARAMS_PN_STIM_overlap" + str(u))
    PARAMS_STIM_DBC_overlap.append("PARAMS_DBC_STIM_overlap" + str(u))
    PARAMS_STIM_PN_overlap[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS
                                 "tauSyn": 0.1, #ms
                                 "rate": 1700,
                                 "STIM_start": (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration),
                                 "STIM_stop": cue_duration + (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration)}
    PARAMS_STIM_DBC_overlap[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS
                                  "tauSyn": 0.1, #ms
                                  "rate": 75,
                                 "STIM_start": (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration),
                                 "STIM_stop": cue_duration + (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration)}


### Controlled overlap
alphabet = []
when_overlap = [0] * (NUM_MC*NUM_HC)
rand = []
start = ord('A')
overlap = []
E=0
F=0
for G in range (NUM_MC*NUM_HC):
    #alphabet.append(chr(start + G))
    alphabet.append(0)
J=0
L=0
N=[]
new_overlap_connection = [] #* (NUM_MC*NUM_HC)
PNx_PNy_overlap_AMPA = []
PNx_PNy_overlap_NMDA = []
PNy_PNx_overlap_AMPA = []
PNy_PNx_overlap_NMDA = []
delay_overlap = 2500

I_want_controlled_overlap = False # can control the overlapping MCs with the lists below when put to "True"
if I_want_controlled_overlap == True:
    rand = [0] * (NUM_MC*NUM_HC)
    this_MC_overlaps = [3, 2] #index starts at 0
    this_MC_overlaps_with = [5, 1] # index starts at 0
    MC_is_in_HC = [0, 4] #index starts at 0
    how_many_MCs_overlap = len(this_MC_overlaps)
    for H in range(how_many_MCs_overlap):
        overlap.append(this_MC_overlaps[H] + NUM_MC * MC_is_in_HC[H])
        rand[overlap[H]] = 100
        alphabet[overlap[H]] = this_MC_overlaps_with[H] - this_MC_overlaps[H]

# initial value current source variable
input_init = {"current": 0.0}

# current source names over HCs
STIM_PN = []
STIM_DBC =[]

#CUEING ITEM0
#parameters
CUE_ITEM_PN_PARAMS = []
CUE_ITEM_DBC_PARAMS = []

for u in range (NUM_MC):
    weight_potentiation = 0 #0 if no cue is wished
    CUE_ITEM_PN_PARAMS.append("PARAMS_PN_STIM" + str(u))
    CUE_ITEM_DBC_PARAMS.append("PARAMS_DBC_STIM" + str(u))
    CUE_ITEM_PN_PARAMS[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS
                             "tauSyn": 0.1, #ms
                             "rate": 1700,
                             "STIM_start": (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration),
                             "STIM_stop": cue_duration + (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration)}
    CUE_ITEM_DBC_PARAMS[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS
                              "tauSyn": 0.1, #ms
                              "rate": 75,
                             "STIM_start": (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration),
                             "STIM_stop": cue_duration + (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration)}

CUE_ITEM_PN = []
CUE_ITEM_DBC_a = []
CUE_ITEM_DBC_b = []
d=0
e=0
f=0

#DISRUPTOR
weight_potentiation = 0 # 0 if no disruptor wished
disruptor_PARAMS = {"weight": PARAMS_STIM_PN[0]["weight"] * weight_potentiation, #nS, 1.5
                    "tauSyn": 0.1, #ms
                    "rate": 1700,
                    "STIM_start": encoding_duration,
                    "STIM_stop": encoding_duration + disruptor_duration}
disruptor = []
fifteen_percent = []
for i in range (NUM_MC*NUM_HC):
    fifteen_percent.append(random.choice(list(range(100))))
    #fifteen_percent[i] = random.choice(list(range(100)))
I=0

for i in range(NUM_HC):  
    # Basket cell between MCs
    BC_HC.append("BC_HC" + str(i))
    BC_HC[i] = model.add_neuron_population(BC_HC[i], NUM_BC, alif, BC_alif_PARAMS, BC_alif_init)
    
    # Turn on spike recording
    BC_HC[i].spike_recording_enabled = True  
    
    for j in range(NUM_MC):
        PN_MC__HC.append("PN" + str(j) + "__HC" + str(i))
        DBC_MC__HC.append("DBC" + str(j) + "__HC" + str(i))
        
        PN_MC_PN_MC__HC_AMPA.append("PN" + str(j) + "_PN" + str(j) + "__HC" + str(i) + "_AMPA")
        PN_MC_PN_MC__HC_NMDA.append("PN" + str(j) + "_PN" + str(j) + "__HC" + str(i) + "_NMDA")
        DBC_MC_PN_MC__HC.append("DBC" + str(j) + "_PN" + str(j) + "__HC" + str(i))
        PN_MC_BC__HC_AMPA.append("PN" + str(j) + "_BC" + str(i) + "__HC" + str(i) + "_AMPA")
        PN_MC_BC__HC_NMDA.append("PN" + str(j) + "_BC" + str(i) + "__HC" + str(i) + "_NMDA")
        BC_PN_MC__HC.append("BC" + str(i) + "_PN" + str(j) + "__HC" + str(i))
        
        #without distractor
        ZMNe_PN_HC.append("ZMNe_PN" + str(j) + "_HC" + str(i))
        ZMNi_PN_HC.append("ZMNi_PN" + str(j) + "_HC" + str(i))
        ZMNe_DBC_HC.append("ZMNe_DBC" + str(j) + "_HC" + str(i))
        ZMNi_DBC_HC.append("ZMNi_DBC" + str(j) + "_HC" + str(i))
        
        #with distractor
        ZMNe_PN_HC_bef_dis.append("ZMNe_bef_dis_PN" + str(j) + "_HC" + str(i))
        ZMNi_PN_HC_bef_dis.append("ZMNi_bef_dis_PN" + str(j) + "_HC" + str(i))
        ZMNe_DBC_HC_bef_dis.append("ZMNe_bef_dis_DBC" + str(j) + "_HC" + str(i))
        ZMNi_DBC_HC_bef_dis.append("ZMNi_bef_dis_DBC" + str(j) + "_HC" + str(i))
        
        ZMNe_PN_HC_dis.append("ZMNe_dis_PN" + str(j) + "_HC" + str(i))
        ZMNi_PN_HC_dis.append("ZMNi_dis_PN" + str(j) + "_HC" + str(i))
        ZMNe_DBC_HC_dis.append("ZMNe_dis_DBC" + str(j) + "_HC" + str(i))
        ZMNi_DBC_HC_dis.append("ZMNi_dis_DBC" + str(j) + "_HC" + str(i))
        
        ZMNe_PN_HC_aft_dis.append("ZMNe_aft_dis_PN" + str(j) + "_HC" + str(i))
        ZMNi_PN_HC_aft_dis.append("ZMNi_aft_dis_PN" + str(j) + "_HC" + str(i))
        ZMNe_DBC_HC_aft_dis.append("ZMNe_aft_dis_DBC" + str(j) + "_HC" + str(i))
        ZMNi_DBC_HC_aft_dis.append("ZMNi_aft_dis_DBC" + str(j) + "_HC" + str(i))
        
        ### NEURON POPULATION
        # MINICOLUMN
        PN_MC__HC[z] = model.add_neuron_population(PN_MC__HC[z], NUM_PN, alif, PN_alif_PARAMS, PN_alif_init)
        DBC_MC__HC[z] = model.add_neuron_population(DBC_MC__HC[z], NUM_DBC, alif, DBC_alif_PARAMS, DBC_alif_init)
    
        # Turn on spike recording
        PN_MC__HC[z].spike_recording_enabled = True
        DBC_MC__HC[z].spike_recording_enabled = True 

    
        ### SYNAPSE POPULATION   
        # MINICOLUMN
        #recurrent excitation pyramidal neurons AMPA synapse
        PN_MC_PN_MC__HC_AMPA[z] = model.add_synapse_population(PN_MC_PN_MC__HC_AMPA[z], "SPARSE_INDIVIDUALG", delay_within_HC, 
                                             PN_MC__HC[z], PN_MC__HC[z],
                                             BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": AMPA_decay}, {},
                                             genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN))
        #recurrent excitatory NMDA synapse between pyramidal neurons
        PN_MC_PN_MC__HC_NMDA[z] = model.add_synapse_population(PN_MC_PN_MC__HC_NMDA[z], "SPARSE_INDIVIDUALG", delay_within_HC,
                                             PN_MC__HC[z], PN_MC__HC[z],
                                             BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": NMDA_decay}, {},
                                             genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN))
        #inhibitory GABA synapse from double bouquet cells to pyramidal neurons (non-plastic)
        DBC_MC_PN_MC__HC[z] = model.add_synapse_population(DBC_MC_PN_MC__HC[z], "DENSE_INDIVIDUALG", delay_within_HC,
                                             DBC_MC__HC[z], PN_MC__HC[z],
                                             i2e_DBC_PN, {}, {"g": -8.0}, {}, {},
                                             "ExpCurr", {"tau": GABA_decay}, {})
        # BETWEEN MINICOLUMNS
        for r in range (1, j+1):
            PN_MCx_DBC_MCy_HC_AMPA.append("PN" + str(j-r) + "_DBC" + str(j) + "__HC" + str(i) + "_AMPA")
            PN_MCy_DBC_MCx_HC_AMPA.append("PN" + str(j) + "_DBC" + str(j-r) + "__HC" + str(i) + "_AMPA")
            PN_MCx_DBC_MCy_HC_NMDA.append("PN" + str(j-r) + "_DBC" + str(j) + "__HC" + str(i) + "_NMDA")
            PN_MCy_DBC_MCx_HC_NMDA.append("PN" + str(j) + "_DBC" + str(j-r) + "__HC" + str(i) + "_NMDA")
            #excitatory AMPA synapse from pyramidal neurons to double bouquet cells of other minicolumns
            PN_MCx_DBC_MCy_HC_AMPA[s] = model.add_synapse_population(PN_MCx_DBC_MCy_HC_AMPA[s], "SPARSE_INDIVIDUALG", delay_within_HC,
                                             PN_MC__HC[j-r], DBC_MC__HC[j],
                                             BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": AMPA_decay}, {},
                                             genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
            PN_MCy_DBC_MCx_HC_AMPA[s] = model.add_synapse_population(PN_MCy_DBC_MCx_HC_AMPA[s], "SPARSE_INDIVIDUALG", delay_within_HC,
                                             PN_MC__HC[j], DBC_MC__HC[j-r],
                                             BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": AMPA_decay}, {},
                                             genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
            # excitatory NMDA synapse from pyramidal neurons to double bouquet cells of other minicolumns
            PN_MCx_DBC_MCy_HC_NMDA[s] = model.add_synapse_population(PN_MCx_DBC_MCy_HC_NMDA[s], "SPARSE_INDIVIDUALG", delay_within_HC,
                                             PN_MC__HC[j-r], DBC_MC__HC[j],
                                             BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": NMDA_decay}, {},
                                             genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
            PN_MCy_DBC_MCx_HC_NMDA[s] = model.add_synapse_population(PN_MCy_DBC_MCx_HC_NMDA[s], "SPARSE_INDIVIDUALG", delay_within_HC,
                                             PN_MC__HC[j], DBC_MC__HC[j-r],
                                             BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                             "ExpCurr", {"tau": NMDA_decay}, {},
                                             genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
            s +=1
        # excitatory AMPA synapse from pyramidal neurons to the basket cells of the hypercolumn (non-plastic)
        PN_MC_BC__HC_AMPA[z] = model.add_synapse_population(PN_MC_BC__HC_AMPA[z], "SPARSE_INDIVIDUALG", delay_within_HC,
                                              PN_MC__HC[z], BC_HC[i],
                                              e2i_PN_BC, {}, {"g": 3.5}, {}, {},
                                              "ExpCurr", {"tau": AMPA_decay}, {},
                                              genn_model.init_connectivity("FixedProbability", prob_PN_BC))
        #inhibitory GABA synapse from the basket cells of the hypercolumn to pyramidal neurons (non-plastic)
        BC_PN_MC__HC[z] = model.add_synapse_population(BC_PN_MC__HC[z], "SPARSE_INDIVIDUALG", delay_within_HC,
                                              BC_HC[i], PN_MC__HC[z],
                                              i2e_BC_PN, {}, {"g": -30.0}, {}, {},
                                              "ExpCurr", {"tau": GABA_decay}, {},
                                              genn_model.init_connectivity("FixedProbability", prob_PN_BC))
        # BETWEEN HYPERCOLUMNS
        for q in range(1, i+1):
            for k in range(NUM_MC):
                if i > 0:
                    if j+k == j:
                        PNx_PNy_between_HCs_AMPA.append("PN" + str(z-(NUM_MC*q)) + "_HC" + str(i-q) + "__PN" + str(z) + "_HC" + str(i) +"_AMPA")
                        PNy_PNx_between_HCs_AMPA.append("PN" + str(z) + "_HC" + str(i) + "__PN" + str(z-(NUM_MC*q)) + "_HC" + str(i-q) + "_AMPA")
                        PNx_PNy_between_HCs_NMDA.append("PN" + str(z-(NUM_MC*q)) + "_HC" + str(i-q) + "__PN" + str(z) + "_HC" + str(i) + "_NMDA")
                        PNy_PNx_between_HCs_NMDA.append("PN" + str(z) + "_HC" + str(i) + "__PN" + str(z-(NUM_MC*q)) + "_HC" + str(i-q) + "_NMDA")  
                        #excitatory AMPA synapses from pyramidal neurons from one hypercolumn to co-active pyramidal neurons in other hypercolumns
                        PNx_PNy_between_HCs_AMPA[m] = model.add_synapse_population(PNx_PNy_between_HCs_AMPA[m], "SPARSE_INDIVIDUALG", delay_between_HC, 
                                                                         PN_MC__HC[z-(NUM_MC*q)], PN_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        PNy_PNx_between_HCs_AMPA[m] = model.add_synapse_population(PNy_PNx_between_HCs_AMPA[m], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], PN_MC__HC[z-(NUM_MC*q)],
                                                                         BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        #excitatory NMDA synapses from pyramidal neurons from one hypercolumn to co-active pyramidal neurons in other hypercolumns
                        PNx_PNy_between_HCs_NMDA[m] = model.add_synapse_population(PNx_PNy_between_HCs_NMDA[m], "SPARSE_INDIVIDUALG", delay_between_HC, 
                                                                         PN_MC__HC[z-(NUM_MC*q)], PN_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        PNy_PNx_between_HCs_NMDA[m] = model.add_synapse_population(PNy_PNx_between_HCs_NMDA[m], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], PN_MC__HC[z-(NUM_MC*q)],
                                                                         BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        m+=1
                    elif ((z-j)/i)-k-j > 0:
                        PNx_DBCy_between_HCs_a_AMPA.append("PN" + str(z-j-(NUM_MC*(q-1))-k) + "_HC" + str(i-q) + "__DBC" + str(z) + "_HC" + str(i) + "AMPA")
                        PNy_DBCx_between_HCs_a_AMPA.append("PN" + str(z) + "_HC" + str(i) + "__DBC" + str(z-j-(NUM_MC*(q-1))-k) + "_HC" + str(i-q) + "AMPA")
                        PNx_DBCy_between_HCs_a_NMDA.append("PN" + str(z-j-(NUM_MC*(q-1))-k) + "_HC" + str(i-q) + "__DBC" + str(z) + "_HC" + str(i) + "NMDA")
                        PNy_DBCx_between_HCs_a_NMDA.append("PN" + str(z) + "_HC" + str(i) + "__DBC" + str(z-j-(NUM_MC*(q-1))-k) + "_HC" + str(i-q) + "NMDA")
                        # excitatory AMPA synapses from pyramidal cells to double bouquet cells of competing minicolumns of other hypercolumns
                        PNx_DBCy_between_HCs_a_AMPA[n] = model.add_synapse_population(PNx_DBCy_between_HCs_a_AMPA[n], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z-j-(NUM_MC*(q-1))-k], DBC_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        PNy_DBCx_between_HCs_a_AMPA[n] = model.add_synapse_population(PNy_DBCx_between_HCs_a_AMPA[n], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], DBC_MC__HC[z-j-(NUM_MC*(q-1))-k],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        # excitatory NMDA synapses from pyramidal cells to double bouquet cells of competing minicolumns of other hypercolumns
                        PNx_DBCy_between_HCs_a_NMDA[n] = model.add_synapse_population(PNx_DBCy_between_HCs_a_NMDA[n], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z-j-(NUM_MC*(q-1))-k], DBC_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        PNy_DBCx_between_HCs_a_NMDA[n] = model.add_synapse_population(PNy_DBCx_between_HCs_a_NMDA[n], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], DBC_MC__HC[z-j-(NUM_MC*(q-1))-k],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        n+=1
                    else:
                        PNx_DBCy_between_HCs_b_AMPA.append("PN" + str(z-j-(NUM_MC*q)+((NUM_MC-1)-k)) + "_HC" + str(i-q) + "__DBC" + str(z) + "_HC" + str(i) + "_AMPA")
                        PNy_DBCx_between_HCs_b_AMPA.append("PN" + str(z) + "_HC" + str(i) + "__DBC" + str(z-j-(NUM_MC*q)+((NUM_MC-1)-k)) + "_HC" + str(i-q) + "_AMPA")
                        PNx_DBCy_between_HCs_b_NMDA.append("PN" + str(z-j-(NUM_MC*q)+((NUM_MC-1)-k)) + "_HC" + str(i-q) + "__DBC" + str(z) + "_HC" + str(i) + "_NMDA")
                        PNy_DBCx_between_HCs_b_NMDA.append("PN" + str(z) + "_HC" + str(i) + "__DBC" + str(z-j-(NUM_MC*q)+((NUM_MC-1)-k)) + "_HC" + str(i-q) + "_NMDA")
                        # excitatory AMPA synapses from pyramidal cells to double bouquet cells of competing minicolumns of other hypercolumns
                        PNx_DBCy_between_HCs_b_AMPA[o] = model.add_synapse_population(PNx_DBCy_between_HCs_b_AMPA[o], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z-j-(NUM_MC*q)+((NUM_MC-1)-k)], DBC_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        PNy_DBCx_between_HCs_b_AMPA[o] = model.add_synapse_population(PNy_DBCx_between_HCs_b_AMPA[o], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], DBC_MC__HC[z-j-(NUM_MC*q)+((NUM_MC-1)-k)],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        # excitatory NMDA synapses from pyramidal cells to double bouquet cells of competing minicolumns of other hypercolumns
                        PNx_DBCy_between_HCs_b_NMDA[o] = model.add_synapse_population(PNx_DBCy_between_HCs_b_NMDA[o], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z-j-(NUM_MC*q)+((NUM_MC-1)-k)], DBC_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        PNy_DBCx_between_HCs_b_NMDA[o] = model.add_synapse_population(PNy_DBCx_between_HCs_b_NMDA[o], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], DBC_MC__HC[z-j-(NUM_MC*q)+((NUM_MC-1)-k)],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbability", prob_PN_DBC))
                        o+=1
        
        ### ZERO MEAN NOISE     
        #without distractor                 
        ZMNe_PN_HC[z] = model.add_current_source(ZMNe_PN_HC[z], PoissonExp_STIM, PN_MC__HC[z], ZMNe_PN_PARAMS, input_init)
        ZMNi_PN_HC[z] = model.add_current_source(ZMNi_PN_HC[z], PoissonExp_STIM, PN_MC__HC[z], ZMNi_PN_PARAMS, input_init)
        ZMNe_DBC_HC[z] = model.add_current_source(ZMNe_DBC_HC[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNe_DBC_PARAMS, input_init)
        ZMNi_DBC_HC[z] = model.add_current_source(ZMNi_DBC_HC[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNi_DBC_PARAMS, input_init)
        
        #if a distractor for Fig. 4, 5B and 5D is wished, use this ZMN instead
        #ZMNe_PN_HC_bef_dis[z] = model.add_current_source(ZMNe_PN_HC_bef_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNe_PN_PARAMS_bef_dis, input_init)
        #ZMNi_PN_HC_bef_dis[z] = model.add_current_source(ZMNi_PN_HC_bef_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNi_PN_PARAMS_bef_dis, input_init)
        #ZMNe_DBC_HC_bef_dis[z] = model.add_current_source(ZMNe_DBC_HC_bef_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNe_DBC_PARAMS_bef_dis, input_init)
        #ZMNi_DBC_HC_bef_dis[z] = model.add_current_source(ZMNi_DBC_HC_bef_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNi_DBC_PARAMS_bef_dis, input_init)
        
        # DISTRACTOR
        #ZMNe_PN_HC_dis[z] = model.add_current_source(ZMNe_PN_HC_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNe_PN_PARAMS_dis, input_init)
        #ZMNi_PN_HC_dis[z] = model.add_current_source(ZMNi_PN_HC_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNi_PN_PARAMS_dis, input_init)
        #ZMNe_DBC_HC_dis[z] = model.add_current_source(ZMNe_DBC_HC_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNe_DBC_PARAMS_dis, input_init)
        #ZMNi_DBC_HC_dis[z] = model.add_current_source(ZMNi_DBC_HC_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNi_DBC_PARAMS_dis, input_init)
        
        #ZMNe_PN_HC_aft_dis[z] = model.add_current_source(ZMNe_PN_HC_aft_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNe_PN_PARAMS_aft_dis, input_init)
        #ZMNi_PN_HC_aft_dis[z] = model.add_current_source(ZMNi_PN_HC_aft_dis[z], PoissonExp_STIM, PN_MC__HC[z], ZMNi_PN_PARAMS_aft_dis, input_init)
        #ZMNe_DBC_HC_aft_dis[z] = model.add_current_source(ZMNe_DBC_HC_aft_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNe_DBC_PARAMS_aft_dis, input_init)
        #ZMNi_DBC_HC_aft_dis[z] = model.add_current_source(ZMNi_DBC_HC_aft_dis[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNi_DBC_PARAMS_aft_dis, input_init)
        
        
        ### CURRENT SOURCE
        if I_want_controlled_overlap == False:  
            for B in range (NUM_MC):
                C = range(1-B, NUM_MC-B)
                alphabet[NUM_MC*i+B] = random.choice(C)
                if alphabet[NUM_MC*i+B] <= 0:
                    alphabet[NUM_MC*i+B] -=1
            for D in range(NUM_MC):
                rand.append("random_number" + str(E))
                rand[NUM_MC*i+D] = random.randint(0,100)
                E+=1
                        
        for v in range(NUM_MC):
            #no overlap
            if rand[z] <= 100-overlap_percentage:
                if v == j:
                    STIM_PN.append("STIM" + str(j) + "_PN" + str(j) + "__HC" + str(i))
                    STIM_PN[z] = model.add_current_source(STIM_PN[z], PoissonExp_STIM, PN_MC__HC[z], PARAMS_STIM_PN[j], input_init)
                else:
                    STIM_DBC.append("STIM" + str(v) + "_DBC" + str(j) + "__HC" + str(i))
                    STIM_DBC[F] = model.add_current_source(STIM_DBC[F], PoissonExp_STIM, DBC_MC__HC[z], PARAMS_STIM_DBC[v], input_init)
                    F+=1
            #overlap
            else:
                print("Overlap from MC", j, "of HC", i, "to MC", int(j)+int(alphabet[z]))
                #overlap to higher MC 1/2
                if alphabet[z] > 0 and v == j:
                    when_overlap[z+alphabet[z]] = alphabet[z]
                    STIM_PN.append("STIM" + str(j) + "_PN" + str(int(z)-int(when_overlap[z])) + "__HC" + str(i))
                    new_overlap_connection.append(z)
                    N.append(alphabet[z])
                    J+=1
                STIM_DBC.append("STIM" + str(v) + "_DBC" + str(j) + "__HC" + str(i))
                STIM_DBC[F] = model.add_current_source(STIM_DBC[F], PoissonExp_STIM, DBC_MC__HC[z], PARAMS_STIM_DBC[v], input_init)
                F+=1
                #overlap to lower MC 1/1
                if alphabet[z] < 0 and v == j:
                    STIM_PN.append("STIM" + str(j) + "_PN" + str(int(z)+int(alphabet[z])) + "__HC" + str(i))
                    STIM_PN[z] = model.add_current_source(STIM_PN[z], PoissonExp_STIM, PN_MC__HC[int(z)+int(alphabet[z])], PARAMS_STIM_PN[j], input_init)
                    STIM_DBC.pop((F-1)-NUM_MC+(NUM_MC-1)*alphabet[z]-alphabet[z]+(j+alphabet[z]))
                    F-=1
                    #connections to already built HCs
                    for O in range (i):
                        PNx_PNy_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z+alphabet[z]) + "_2_MC" + str(z-(O+1)*NUM_MC) + "of_HC_" + str(i))
                        PNx_PNy_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z+alphabet[z]) + "_2_MC" + str(z-(O+1)*NUM_MC) + "of_HC_" + str(i))
                        PNy_PNx_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z-(O+1)*NUM_MC) + "_2_MC" + str(z+alphabet[z]) + "of_HC_" + str(i))
                        PNy_PNx_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z-(O+1)*NUM_MC) + "_2_MC" + str(z+alphabet[z]) + "of_HC_" + str(i))
                        PNx_PNy_overlap_AMPA[L] = model.add_synapse_population(PNx_PNy_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z+alphabet[z]], PN_MC__HC[z-(O+1)*NUM_MC],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                        PNx_PNy_overlap_NMDA[L] = model.add_synapse_population(PNx_PNy_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z+alphabet[z]], PN_MC__HC[z-(O+1)*NUM_MC],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": NMDA_decay}, {})
                        PNy_PNx_overlap_AMPA[L] = model.add_synapse_population(PNy_PNx_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z-(O+1)*NUM_MC], PN_MC__HC[z+alphabet[z]],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                        PNy_PNx_overlap_NMDA[L] = model.add_synapse_population(PNy_PNx_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z-(O+1)*NUM_MC], PN_MC__HC[z+alphabet[z]],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": NMDA_decay}, {})
                        L+=1
                    new_overlap_connection.append(int(z))
                    N.append(alphabet[z])
            #overlap to higher MC 2/2
            if when_overlap[z] > 0 and v==j:
                STIM_PN[int(int(z)-int(when_overlap[z]))] = model.add_current_source(STIM_PN[int(int(z)-int(when_overlap[z]))], PoissonExp_STIM, PN_MC__HC[z], PARAMS_STIM_PN[j-when_overlap[z]], input_init)
                STIM_DBC.pop((F-1)-(NUM_MC-2)+(j-when_overlap[z]))
                F-=1
                #connections to already built HCs
                for K in range (i):
                    PNx_PNy_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z+alphabet[z]) + "2_MC" + str(z-(K+1)*NUM_MC) + "of_HC_" + str(i))
                    PNx_PNy_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z+alphabet[z]) + "2_MC" + str(z-(K+1)*NUM_MC) + "of_HC_" + str(i))
                    PNy_PNx_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z-(K+1)*NUM_MC) + "2_MC" + str(z+alphabet[z]) + "of_HC_" + str(i))
                    PNy_PNx_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z-(K+1)*NUM_MC) + "2_MC" + str(z+alphabet[z]) + "of_HC_" + str(i))
                    PNx_PNy_overlap_AMPA[L] = model.add_synapse_population(PNx_PNy_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z+alphabet[z]], PN_MC__HC[z-(K+1)*NUM_MC],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                    PNx_PNy_overlap_NMDA[L] = model.add_synapse_population(PNx_PNy_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z+alphabet[z]], PN_MC__HC[z-(K+1)*NUM_MC],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                    PNy_PNx_overlap_AMPA[L] = model.add_synapse_population(PNy_PNx_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z-(K+1)*NUM_MC], PN_MC__HC[z+alphabet[z]],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                    PNy_PNx_overlap_NMDA[L] = model.add_synapse_population(PNy_PNx_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                    PN_MC__HC[z-(K+1)*NUM_MC], PN_MC__HC[z+alphabet[z]],
                                                                    BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                    "ExpCurr", {"tau": AMPA_decay}, {})
                    L+=1
            #connections to MCs of HCs to come
            if len(new_overlap_connection) != 0:
                for M in range(len(new_overlap_connection)):
                    if (int(new_overlap_connection[M])-j) % NUM_MC == 0 and z != new_overlap_connection[M] and v == j:
                        PNx_PNy_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(int(new_overlap_connection[M])+int(N[M])) + "_2_MC" + str(z) + "of_HC_" + str(i))
                        PNx_PNy_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(int(new_overlap_connection[M])+int(N[M])) + "_2_MC" + str(z) + "of_HC_" + str(i))
                        PNy_PNx_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z) + "_2_MC" + str(int(new_overlap_connection[M])+int(N[M])) + "of_HC_" + str(i))
                        PNy_PNx_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z) + "_2_MC" + str(int(new_overlap_connection[M])+int(N[M])) + "of_HC_" + str(i))
                        PNx_PNy_overlap_AMPA[L] = model.add_synapse_population(PNx_PNy_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                            PN_MC__HC[int(new_overlap_connection[M])+int(N[M])], PN_MC__HC[z],
                                                                            BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                            "ExpCurr", {"tau": AMPA_decay}, {})
                        PNx_PNy_overlap_NMDA[L] = model.add_synapse_population(PNx_PNy_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                            PN_MC__HC[int(new_overlap_connection[M])+int(N[M])], PN_MC__HC[z],
                                                                            BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                            "ExpCurr", {"tau": AMPA_decay}, {})
                        PNy_PNx_overlap_AMPA[L] = model.add_synapse_population(PNy_PNx_overlap_AMPA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                            PN_MC__HC[z], PN_MC__HC[int(new_overlap_connection[M])+int(N[M])],
                                                                            BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                            "ExpCurr", {"tau": AMPA_decay}, {})
                        PNy_PNx_overlap_NMDA[L] = model.add_synapse_population(PNy_PNx_overlap_NMDA[L], "DENSE_INDIVIDUALG", delay_overlap, 
                                                                            PN_MC__HC[z], PN_MC__HC[int(new_overlap_connection[M])+int(N[M])],
                                                                            BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                            "ExpCurr", {"tau": AMPA_decay}, {})
                        L+=1
                

        ### CUEING ITEM(S)
        if j >= 0 and j < NUM_MC-0: #decide what item(s) by scaling the two values now set to 0
            for c in range(NUM_MC-0): #decide how many items by scaling value now set to 0
                if c == 0:
                    CUE_ITEM_PN.append("CUE_ITEM_PN" + str(z) + "__HC" + str(i))
                    CUE_ITEM_PN[f] = model.add_current_source(CUE_ITEM_PN[f], PoissonExp_STIM, PN_MC__HC[z], CUE_ITEM_PN_PARAMS[j], input_init)
                    f+=1
                elif j-c >= 0:
                    CUE_ITEM_DBC_a.append("DBC" + str(z) + "__HC" + str(i) + "CUE_ITEM" + str(j-c))
                    CUE_ITEM_DBC_a[d] = model.add_current_source(CUE_ITEM_DBC_a[d], PoissonExp_STIM, DBC_MC__HC[z], CUE_ITEM_DBC_PARAMS[j-c], input_init)
                    d+=1
                else:
                    CUE_ITEM_DBC_b.append("DBC" + str(z) + "__HC" + str(i) + "CUE_ITEM" + str(c))
                    CUE_ITEM_DBC_b[e] = model.add_current_source(CUE_ITEM_DBC_b[e], PoissonExp_STIM, DBC_MC__HC[z], CUE_ITEM_DBC_PARAMS[c], input_init)
                    e+=1
        z+=1
print("It works, good job!")


model._model.set_fuse_postsynaptic_models(True)
model._model.set_fuse_pre_post_weight_update_models(True)

# Build model and load it
model.build()
duration_timestep = int(duration_time / DT)
model.load(num_recording_timesteps = duration_timestep)

### Record voltage, weight, current
c_PN0 = np.empty((50000, 30))
c_DBC1 = np.empty((50000, 1))
c_PN1 = np.empty((50000, 30))
c_DBC0 = np.empty((50000, 1))

#empty lists for voltage values
v_PN_MC__HC = []
v_DBC_MC__HC = []
v_BC_HC = []

PN_MC__HC_v_view=[]
DBC_MC__HC_v_view=[]
BC_HC_v_view=[]

#empty lists for synaptic weight values
g_PN_MCx_DBC_MCy_HC_AMPA =[]
g_PNy_PNx_between_HCs_AMPA=[]
g_PN_MCx_DBC_MCy_HC_NMDA =[]
g_PNx_PNy_between_HCs_NMDA=[]
g_PNx_PNy_overlap=[]

PN_MCx_DBC_MCy_HC_AMPA_g_view=[]
PNy_PNx_between_HCs_AMPA_g_view=[]
PN_MCx_DBC_MCy_HC_NMDA_g_view=[]
PNx_PNy_between_HCs_NMDA_g_view=[]
PNx_PNy_overlap_g_view=[]

#empty lists for BCPNN values
PNx_PNy_between_HCs_AMPA_Zi_view=[]
PNx_PNy_between_HCs_AMPA_Zj_view=[]
PNx_PNy_between_HCs_AMPA_Pi_view=[]
PNx_PNy_between_HCs_AMPA_Pj_view=[]
PNx_PNy_between_HCs_AMPA_Pij_view=[]

for i in range(NUM_MC):
    #append empty arrays to voltage lists
    v_PN_MC__HC.append(np.empty((int(duration_time/DT), NUM_PN)))
    v_DBC_MC__HC.append(np.empty((int(duration_time/DT), NUM_DBC)))
    
    #append empty arrays to synaptic weight lists
    g_PN_MCx_DBC_MCy_HC_AMPA.append(np.empty((int(duration_time/DT), NUM_PN*NUM_DBC)))
    g_PNy_PNx_between_HCs_AMPA.append(np.empty((int(duration_time/DT), 510)))
    g_PNx_PNy_between_HCs_NMDA.append(np.empty((int(duration_time/DT), 510)))
    g_PNx_PNy_overlap.append(np.empty((int(duration_time/DT), 900)))
    
    #append voltage values to list
    PN_MC__HC_v_view.append(PN_MC__HC[i].vars["V"].view)
    DBC_MC__HC_v_view.append(DBC_MC__HC[i].vars["V"].view)
    
    #append synaptic weight values to list
    PN_MCx_DBC_MCy_HC_AMPA_g_view.append(PN_MCx_DBC_MCy_HC_AMPA[i].vars["g"].view)
    PNy_PNx_between_HCs_AMPA_g_view.append(PNy_PNx_between_HCs_AMPA[i].vars["g"].view)
    PNx_PNy_between_HCs_NMDA_g_view.append(PNx_PNy_between_HCs_NMDA[i].vars["g"].view)
    
for j in range(NUM_HC):
    v_BC_HC.append(np.empty((int(duration_time/DT), NUM_BC)))   
    BC_HC_v_view.append(BC_HC[j].vars["V"].view)
    
while model.t < duration_time:
    model.step_time()

    for i in range(NUM_MC):
        PN_MC__HC[i].pull_var_from_device("V")
        DBC_MC__HC[i].pull_var_from_device("V")
        PN_MCx_DBC_MCy_HC_AMPA[i].pull_var_from_device("g")
        PNy_PNx_between_HCs_AMPA[i].pull_var_from_device("g")
        PNx_PNy_between_HCs_NMDA[i].pull_var_from_device("g")
    
    for i in range(NUM_HC):
        BC_HC[i].pull_var_from_device("V")
    
    for i in range(NUM_MC):
        v_PN_MC__HC[i][model.timestep - 1,:]=PN_MC__HC_v_view[i][:]
        v_DBC_MC__HC[i][model.timestep - 1,:]=DBC_MC__HC_v_view[i][:]
        
        g_PN_MCx_DBC_MCy_HC_AMPA[i][model.timestep - 1,:]=PN_MCx_DBC_MCy_HC_AMPA_g_view[i][:]
        g_PNy_PNx_between_HCs_AMPA[i][model.timestep - 1,:]=PNy_PNx_between_HCs_AMPA_g_view[i][:]
        g_PNx_PNy_between_HCs_NMDA[i][model.timestep - 1,:]=PNx_PNy_between_HCs_NMDA_g_view[i][:]
        
    for i in range(NUM_HC):
        v_BC_HC[i][model.timestep - 1,:]=BC_HC_v_view[i][:]
        
#extract spike times and neuron ID
model.pull_recording_buffers_from_device()

# measure time different parts of the network take to simulate
sim_start_time = perf_counter()
sim_end_time =  perf_counter()

print("Timing:")
print("\tSimulation:%f" % ((sim_end_time - sim_start_time) * 1000.0))

if MEASURE_TIMING:
    print("\tInit:%f" % (1000.0 * model.init_time))
    print("\tSparse init:%f" % (1000.0 * model.init_sparse_time))
    print("\tNeuron simulation:%f" % (1000.0 * model.neuron_update_time))
    print("\tSynapse simulation:%f" % (1000.0 * model.presynaptic_update_time))

# record overlap percentage
plot = 0
ref = [0] * NUM_MC
for i in range(NUM_MC):
    which_neuron = v_PN_MC__HC[i]
    for k in range(int(encoding_duration/DT), len(which_neuron)):
        for l in range(len(which_neuron[k])):
            if which_neuron[k][l] < -90 and ref[i] == 0: #check if -90 if too low for overlapping MCs
                plot += 1/NUM_MC
                ref[i]+=1
#values saved to respective file
with open("Capacity_example.txt", "a+") as output:
    output.write(str(plot))
    output.write(" ")
 
# record synaptic weights
# weights between MC0 of HC0 and HC1
ooi = g_PNx_PNy_between_HCs_NMDA[0][-1]
ooi_averaged = sum(g_PNx_PNy_between_HCs_NMDA[0][-1]) / len(g_PNx_PNy_between_HCs_NMDA[0][-1])
ooi = np.delete(ooi, np.where(ooi == 0))
ooi_nz_averaged = sum(ooi) / len(ooi)       
        
with open("Synaptic_weights_MC0_example.txt", "a+") as output:
    output.write(str(ooi_averaged))
    output.write(" ")
    
#exclude non-activated synapses (since connection probability is set to 20%)
with open("Synaptic_weights_MC0_without0_example", "a+") as output:
    output.write(str(ooi_nz_averaged))
    output.write(" ")

#weights between all co-active MCs o all HCs
for i in range(len(g_PNx_PNy_between_HCs_NMDA)):
    all_MC = g_PNx_PNy_between_HCs_NMDA[i][-1]
    all_MC_nz = np.delete(all_MC, np.where(all_MC == 0))
    all_MC_nz_average = np.delete(all_MC, np.where(all_MC == 0))
    all_MC_nz_averaged = sum(all_MC_nz_average) / len(all_MC_nz_average)
    all_MC_nz_averages.append(all_MC_nz_averaged)
    all_MC_nz_averages_averaged = sum(all_MC_nz_averages) / len(all_MC_nz_averages)
    
    all_MC_averaged = sum(g_PNx_PNy_between_HCs_NMDA[i][-1]) / len(g_PNx_PNy_between_HCs_NMDA[i][-1])
    all_MC_averages.append(all_MC_averaged)
    all_MC_averages_averaged = sum(all_MC_averages) / len(all_MC_averages)
    
with open("Synaptic_weights_allMC_example.txt", "a+") as output:
    output.write(str(all_MC_averages_averaged))
    output.write(" ")

#exclude non-activated synapses
with open("Synaptic_weights_allMC_without0_example.txt", "a+") as output:
    output.write(str(all_MC_nz_averages_averaged))
    output.write(" ")

PN_spike_times=[]
PN_spike_ids=[]
DBC_spike_times=[]
DBC_spike_ids=[]
BC_spike_times=[]
BC_spike_ids=[]
all_PN_spikes=[]
all_DBC_spikes=[]
spike_frequency_PN=[]
spike_frequency_DBC=[]

fig, axes = plt.subplots(2, sharex=True)

# plot 
i=0
how_many_HC_shown = 2
which_HC_shown = 0 #index starts at 0
for k in range(how_many_HC_shown):
    for j in range(NUM_MC*which_HC_shown,NUM_MC*(which_HC_shown+1)): #just show HC0, rest "reflected" in it
        PN_spike_times.append("spike_times_PN" + str(i))
        PN_spike_ids.append("spike_ids_PN" + str(i))
        DBC_spike_times.append("spike_times_DBC" + str(i))
        DBC_spike_ids.append("spike_ids_DBC" + str(i))
        BC_spike_times.append("spike_times_BC" + str(i))
        BC_spike_ids.append("spike_ids_BC" + str(i))
        all_PN_spikes.append("all_spikes_PN_MC" + str(i))
        all_DBC_spikes.append("all_spikes_DBC_MC" + str(i))
        spike_frequency_PN.append("spike_frequency_PN_MC" + str(i))
        spike_frequency_DBC.append("spike_frequency_DBC_MC" + str(i))
        PN_spike_times[i], PN_spike_ids[i] = PN_MC__HC[j+(k*NUM_MC)].spike_recording_data 
        DBC_spike_times[i], DBC_spike_ids[i] = DBC_MC__HC[j+(k*NUM_MC)].spike_recording_data
        BC_spike_times[i], BC_spike_ids[i] = BC_HC[which_HC_shown].spike_recording_data
        if i < NUM_MC:
            axes[1].scatter(PN_spike_times[i]/DT, PN_spike_ids[i] + NUM_PN*i, s=1)
        else:
            axes[0].scatter(PN_spike_times[i]/DT, PN_spike_ids[i] + NUM_PN*i, s=1)
            
        # can be used to measure frequency 
        #all_PN_spikes[i] = len(PN_spike_times[i])/NUM_PN
        #all_DBC_spikes[i] = len(DBC_spike_times[i])/NUM_DBC
        #spike_frequency_PN[i] = all_PN_spikes[i] / duration_time_sec
        #spike_frequency_DBC[i] = all_DBC_spikes[i] / duration_time_sec
        i+=1

#used to plot weights in Fig. 5
#axes[2].plot(g_PNx_PNy_between_HCs_NMDA[0])

### Customize Axes
# x-axis in seconds
duration_time_rounded=round(duration_time,-3)
plt.xticks([i*10000 for i in range(int(((duration_time_rounded/1000)+1)))], [str(i) for i in range(int(((duration_time_rounded/1000)+1)))])

# y-axis as MCs
MC_name=[]
for i in range(NUM_MC):
    MC_name.append("MC" + str(i))
axes[0].set_yticks(np.arange(NUM_PN*NUM_MC+NUM_PN/2, NUM_PN*NUM_MC+(NUM_MC+1)*NUM_PN-(NUM_PN-2), NUM_PN))
axes[0].set_yticklabels(MC_name)
axes[1].set_yticks(np.arange(NUM_PN/2, (NUM_MC+1)*NUM_PN-(NUM_PN-2), NUM_PN))
axes[1].set_yticklabels(MC_name)

### label axes
axes[0].title.set_text("HC1")
axes[0].axvspan(0, encoding_duration/DT, color='m', alpha=0.1)

# with disruptor
pre_free_recall = encoding_duration/DT + disruptor_duration/DT
#axes[0].axvspan(encoding_duration/DT, pre_free_recall, color='y', alpha=0.1)
#axes[0].axvspan(pre_free_recall, duration_timestep, color='c', alpha=0.1)

# without disruptor
axes[0].axvspan(encoding_duration/DT, duration_timestep, color='c', alpha=0.1)
axes[1].axvspan(0, encoding_duration/DT, color='m', alpha=0.1, label='encoding phase')
axes[1].title.set_text("HC0")
axes[1].set_xlabel("Time [s]")

# with disruptor
#axes[1].axvspan(encoding_duration/DT, pre_free_recall, color='y', alpha=0.1, label='disruptor')
#axes[1].axvspan(pre_free_recall, duration_timestep, color='c', alpha=0.1, label='free recall')

# without disruptor
axes[1].axvspan(encoding_duration/DT, duration_timestep, color='c', alpha=0.1, label='free recall')
axes[1].legend(loc="upper left", fontsize=9)

plt.savefig('outcome.jpg')# bbox_inches='tight')
plt.show()