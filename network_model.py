import math
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import random
from pygenn import genn_model

# How many HC / MC
NUM_HC = 7
NUM_MC = 4

#Simulation time step
DT = 0.1 #ms

### Simulating input
#encoding phase
activation_duration = 300.0 #ms, 200, 510, 300
delay_duration = 400.0 #ms, 500, 400
encoding_duration = (activation_duration + delay_duration) * NUM_MC
#free recall phase
free_recall_duration = 13000.0 #ms
cue_duration = 20.0 #ms
#Stiulation duration
duration_time = (activation_duration + delay_duration) * NUM_MC + free_recall_duration
duration_time_sec = duration_time/1000

#overlap
overlap_percentage = 0.1 #can not be 0.0 for controlled overlap

#for results, overlap implementation worked if the overlapping ones fire after each other in free recall

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
                    "b": 86.0}#86.0, 15

DBC_alif_PARAMS = {"tauM": 20.0,
                    "R_m": 660.0, #660, 1333
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
                "wGain": 1.0} #nS, AMPA 0.76 NMDA 0.07

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
                    ["Epsilon", genn_model.create_dpf_class(lambda pars, dt: pars[2] / (pars[3] * pars[2]))()]], #if ($(t) >= $(encoding_duration)) {
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
stdp_additive_AMPA_PARAMS = {"tauPlus": 20.0,#ms
                            "tauMinus": 20.0,
                            "aPlus": 0.1, #strength of potentiation, niko 1, 0.1
                            "aMinus": 0.15,#strength of depression niko 1, 0.15
                            "alpha": 1, #asymmetry parameter
                            "wMin": 0.0,
                            "wMax": 1000}#13.5
stdp_additive_NMDA_PARAMS = {"tauPlus": 20.0,#ms
                            "tauMinus": 20.0,
                            "aPlus": 0.1, #strength of potentiation, niko 1, 0.1
                            "aMinus": 0.15,#strength of depression niko 1, 0.15
                            "alpha": 1, #asymmetry parameter, 1,2
                            "wMin": 0.0,
                            "wMax": 1000}#3.5

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
    """) #gennrand_uniform: random number between 0 and 1
        
# Create model
model = genn_model.GeNNModel("float", "Hypercolum")
model.dT = DT

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
                  "PijStar": 0.0} #(f_desired/f_max)**2
BCPNN_pre_var_init = {"ZiStar": 0.0,
                      "PiStar": 0.0} #f_desired/f_max
BCPNN_post_var_init = {"ZjStar": 0.0,
                       "PjStar": 0.0} #f_desired/f_max
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

# PARAMETERS ZERO MEAN NOISE
ZMNe_PN_PARAMS = {"weight": 1300, #nS, nikolas 1.5, 1100 too little
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNi_PN_PARAMS = {"weight": -1300, #nS, nikolas -1.5
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNe_DBC_PARAMS = {"weight": 200, #nS, nikolas 0.12
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}
ZMNi_DBC_PARAMS = {"weight": -200, #nS, nikolas -0.12
                  "tauSyn": 0.1, #ms
                  "rate": 750,
                  "STIM_start": 0.0,
                  "STIM_stop": duration_time/DT}

# zmn neuron / synapse pop names over HCs
ZMNe_PN_HC = []
ZMNi_PN_HC = []
ZMNe_DBC_HC = []
ZMNi_DBC_HC = []


# PARAMETERS CURRENT SOURCE
PARAMS_STIM_PN = []
PARAMS_STIM_DBC = []
#w=0
#x=0

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
    
#overlap current
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

### Controlled overlap
I_want_controlled_overlap = True
if I_want_controlled_overlap == True:
    rand = [0] * (NUM_MC*NUM_HC)
    this_MC_overlaps = [0,2] #index starts at 0
    this_MC_overlaps_with = [3,3] # index starts at 0
    MC_is_in_HC = [1,5] #index starts at 0
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
    weight_potentiation = 0
    CUE_ITEM_PN_PARAMS.append("PARAMS_PN_STIM" + str(u))
    CUE_ITEM_DBC_PARAMS.append("PARAMS_DBC_STIM" + str(u))
    CUE_ITEM_PN_PARAMS[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS, 1.5
                             "tauSyn": 0.1, #ms
                             "rate": 1700,
                             "STIM_start": (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration),
                             "STIM_stop": cue_duration + (NUM_MC*(activation_duration + delay_duration) + delay_duration) + u*(cue_duration + delay_duration)}
    CUE_ITEM_DBC_PARAMS[u] = {"weight": PARAMS_STIM_PN[u]["weight"] * weight_potentiation, #nS, 0.8
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
#=0


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
        
        ZMNe_PN_HC.append("ZMNe_PN" + str(j) + "_HC" + str(i))
        ZMNi_PN_HC.append("ZMNi_PN" + str(j) + "_HC" + str(i))
        ZMNe_DBC_HC.append("ZMNe_DBC" + str(j) + "_HC" + str(i))
        ZMNi_DBC_HC.append("ZMNi_DBC" + str(j) + "_HC" + str(i))
        
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
                                             BCPNN, BCPNN_PARAMS, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,#stdp_additive, stdp_additive_AMPA_PARAMS, stdp_additive_var_init
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
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        PNy_PNx_between_HCs_AMPA[m] = model.add_synapse_population(PNy_PNx_between_HCs_AMPA[m], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], PN_MC__HC[z-(NUM_MC*q)],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": AMPA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        #excitatory NMDA synapses from pyramidal neurons from one hypercolumn to co-active pyramidal neurons in other hypercolumns
                        PNx_PNy_between_HCs_NMDA[m] = model.add_synapse_population(PNx_PNy_between_HCs_NMDA[m], "SPARSE_INDIVIDUALG", delay_between_HC, 
                                                                         PN_MC__HC[z-(NUM_MC*q)], PN_MC__HC[z],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
                                                                         "ExpCurr", {"tau": NMDA_decay}, {},
                                                                         genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN_bc))
                        PNy_PNx_between_HCs_NMDA[m] = model.add_synapse_population(PNy_PNx_between_HCs_NMDA[m], "SPARSE_INDIVIDUALG", delay_between_HC,
                                                                         PN_MC__HC[z], PN_MC__HC[z-(NUM_MC*q)],
                                                                         BCPNN, BCPNN_PARAMS_bh, BCPNN_var_init, BCPNN_pre_var_init, BCPNN_post_var_init,
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
        ZMNe_PN_HC[z] = model.add_current_source(ZMNe_PN_HC[z], PoissonExp_STIM, PN_MC__HC[z], ZMNe_PN_PARAMS, input_init)
        ZMNi_PN_HC[z] = model.add_current_source(ZMNi_PN_HC[z], PoissonExp_STIM, PN_MC__HC[z], ZMNi_PN_PARAMS, input_init)
        ZMNe_DBC_HC[z] = model.add_current_source(ZMNe_DBC_HC[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNe_DBC_PARAMS, input_init)
        ZMNi_DBC_HC[z] = model.add_current_source(ZMNi_DBC_HC[z], PoissonExp_STIM, DBC_MC__HC[z], ZMNi_DBC_PARAMS, input_init)
        
        
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
                print("Overlap from MC", j, "of HC", i, "to MC", int(j)+int(alphabet[j]))
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
                    STIM_PN[z] = model.add_current_source(STIM_PN[z], PoissonExp_STIM, PN_MC__HC[int(z)+int(alphabet[z])], PARAMS_STIM_PN[j], input_init)#problemchen
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
                                                                    #genn_model.init_connectivity("FixedProbabilityNoAutapse", prob_PN_PN))
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
                    PNx_PNy_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(str(z)+str(alphabet[z])) + "2_MC" + str(z-(K+1)*NUM_MC) + "of_HC_" + str(i))
                    PNx_PNy_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(str(z)+str(alphabet[z])) + "2_MC" + str(z-(K+1)*NUM_MC) + "of_HC_" + str(i))
                    PNy_PNx_overlap_AMPA.append("AMPA_Overlap_from_MC" + str(z-(K+1)*NUM_MC) + "2_MC" + str(str(z)+str(alphabet[z])) + "of_HC_" + str(i))
                    PNy_PNx_overlap_NMDA.append("NMDA_Overlap_from_MC" + str(z-(K+1)*NUM_MC) + "2_MC" + str(str(z)+str(alphabet[z])) + "of_HC_" + str(i))
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
        if j >= 0 and j < NUM_MC-3: #decide what item(s)
            for c in range(NUM_MC-3): #decide how many items
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

# Build model and load it
model.build()
duration_timestep = int(duration_time / DT)
model.load(num_recording_timesteps = duration_timestep)