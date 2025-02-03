import yaml
from urllib.request import urlopen

url_syst = "https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/systematics.yaml"

syst_dict = yaml.load(
    urlopen(url_syst), yaml.FullLoader
    )

# sum weights variations
def get_sum_weights_dict(url_sumw="https://raw.githubusercontent.com/zctao/ntuplerTT/master/configs/datasets/ttdiffxs382/sumWeights_variations.yaml"):

    try:
        sumWeights_variations_d = yaml.load(urlopen(url_sumw), yaml.FullLoader)
    except:
        sumWeights_variations_d = {}

    return sumWeights_variations_d

# For now, taken from https://gitlab.cern.ch/ttbarDiffXs13TeV/pyTTbarDiffXs13TeV/-/blob/DM_ljets_resolved/python/MC_variations.py
# Check from the sumWeights file instead?
gen_weights_dict = {
    "nominal":" nominal "," nominal ":0,
    "scale_muF_up":" muR = 1.0, muF = 2.0 "," muR = 1.0, muF = 2.0 ":1,
    "scale_muF_down":" muR = 1.0, muF = 0.5 "," muR = 1.0, muF = 0.5 ":2,
    "scale_muR_up":" muR = 2.0, muF = 1.0 "," muR = 2.0, muF = 1.0 ":3,
    "scale_muR_down":" muR = 0.5, muF = 1.0 "," muR = 0.5, muF = 1.0 ":4,
    "PDF4LHC15_0":" PDF set = 90900 "," PDF set = 90900 ":11,
    "PDF4LHC15_1":" PDF set = 90901 "," PDF set = 90901 ":115,
    "PDF4LHC15_2":" PDF set = 90902 "," PDF set = 90902 ":116,
    "PDF4LHC15_3":" PDF set = 90903 "," PDF set = 90903 ":117,
    "PDF4LHC15_4":" PDF set = 90904 "," PDF set = 90904 ":118,
    "PDF4LHC15_5":" PDF set = 90905 "," PDF set = 90905 ":119,
    "PDF4LHC15_6":" PDF set = 90906 "," PDF set = 90906 ":120,
    "PDF4LHC15_7":" PDF set = 90907 "," PDF set = 90907 ":121,
    "PDF4LHC15_8":" PDF set = 90908 "," PDF set = 90908 ":122,
    "PDF4LHC15_9":" PDF set = 90909 "," PDF set = 90909 ":123,
    "PDF4LHC15_10":" PDF set = 90910 "," PDF set = 90910 ":124,
    "PDF4LHC15_11":" PDF set = 90911 "," PDF set = 90911 ":125,
    "PDF4LHC15_12":" PDF set = 90912 "," PDF set = 90912 ":126,
    "PDF4LHC15_13":" PDF set = 90913 "," PDF set = 90913 ":127,
    "PDF4LHC15_14":" PDF set = 90914 "," PDF set = 90914 ":128,
    "PDF4LHC15_15":" PDF set = 90915 "," PDF set = 90915 ":129,
    "PDF4LHC15_16":" PDF set = 90916 "," PDF set = 90916 ":130,
    "PDF4LHC15_17":" PDF set = 90917 "," PDF set = 90917 ":131,
    "PDF4LHC15_18":" PDF set = 90918 "," PDF set = 90918 ":132,
    "PDF4LHC15_19":" PDF set = 90919 "," PDF set = 90919 ":133,
    "PDF4LHC15_20":" PDF set = 90920 "," PDF set = 90920 ":134,
    "PDF4LHC15_21":" PDF set = 90921 "," PDF set = 90921 ":135,
    "PDF4LHC15_22":" PDF set = 90922 "," PDF set = 90922 ":136,
    "PDF4LHC15_23":" PDF set = 90923 "," PDF set = 90923 ":137,
    "PDF4LHC15_24":" PDF set = 90924 "," PDF set = 90924 ":138,
    "PDF4LHC15_25":" PDF set = 90925 "," PDF set = 90925 ":139,
    "PDF4LHC15_26":" PDF set = 90926 "," PDF set = 90926 ":140,
    "PDF4LHC15_27":" PDF set = 90927 "," PDF set = 90927 ":141,
    "PDF4LHC15_28":" PDF set = 90928 "," PDF set = 90928 ":142,
    "PDF4LHC15_29":" PDF set = 90929 "," PDF set = 90929 ":143,
    "PDF4LHC15_30":" PDF set = 90930 "," PDF set = 90930 ":144,
    "isr_alphaS_Var3cUp":"Var3cUp","Var3cUp":193,
    "isr_alphaS_Var3cDown":"Var3cDown","Var3cDown":194,
    "fsr_muR_up":"isr:muRfac=1.0_fsr:muRfac=2.0","isr:muRfac=1.0_fsr:muRfac=2.0":198,
    "fsr_muR_down":"isr:muRfac=1.0_fsr:muRfac=0.5","isr:muRfac=1.0_fsr:muRfac=0.5":199,
}

def get_gen_weight_index(syst_name):
    if not syst_name in gen_weights_dict:
        raise KeyError(f"systematics: unknown generator weight: {syst_name}")

    weight_name = gen_weights_dict[syst_name]
    weight_index = gen_weights_dict[weight_name]
    return weight_index

def select_systematics(name, keywords):
    if keywords:
        for kw in keywords:
            if kw in name:
                return True
        return False
    else:
        return True

# A helper function that returns a list of systematic uncertainty names
def get_systematics(
    name_filters = [], # list of str; Strings for matching and selecting systematic uncertainties. If empty, take all that are available
    syst_type = None, # str; Required systematic uncertainty type e.g. 'Branch' or 'ScaleFactor'. No requirement if None
    list_of_tuples = False, # bool; If True, return a list of tuples that groups the variations of the same systematic uncertainty together: [(syst1_up,syst1_down), (syst2_up, syst2_down), ...]; Otherwise, return a list of strings
    get_weight_types = False, # bool; If True, also return the associated list of weight types
    ):

    if isinstance(name_filters, str):
        name_filters = [name_filters]

    syst_list = []
    wtype_list = []

    # loop over syst_dict
    for k in syst_dict:
        stype = syst_dict[k]['type']

        prefix = syst_dict[k].get('prefix','')

        uncertainties = syst_dict[k].get('uncertainties', [""])

        variations = syst_dict[k].get('variations', [""])

        if syst_type is not None and stype != syst_type:
            continue

        for s in uncertainties:

            if isinstance(s, dict):
                # e.g. {'eigenvars_B': 9}
                # A vector of uncertainties
                assert(len(s)==1)
                sname, vector_length = list(s.items())[0]

                for i in range(vector_length):

                    systs, wtypes = [], []
                    for v in variations:
                        syst_var = '_'.join(filter(None, [prefix, f"{sname}{i+1}", f"{v}"]))

                        if not select_systematics(syst_var, name_filters):
                            continue

                        wtype_var = "nominal"
                        if stype == "ScaleFactor":
                            wtype_var = '_'.join(filter(None, ["weight", prefix, sname, f"{v}", f"{i}"]))
                        elif stype == "GenWeight":
                            raise RuntimeError("GenWeight with uncertainties as dict not supported")

                        systs.append(syst_var)
                        wtypes.append(wtype_var)

                    if list_of_tuples and systs:
                        syst_list.append(tuple(systs))
                        wtype_list.append(tuple(wtypes))
                    else:
                        syst_list += systs
                        wtype_list += wtypes
            else:

                systs, wtypes = [], []
                for v in variations:
                    syst_var = '_'.join(filter(None, [prefix, f"{s}", f"{v}"]))

                    if not select_systematics(syst_var, name_filters):
                        continue

                    wtype_var = "nominal"
                    if stype == "ScaleFactor":
                        wtype_var = f"weight_{syst_var}"
                    elif stype == "GenWeight":
                        wtype_var = f"mc_generator_weights_{syst_var}:{get_gen_weight_index(syst_var)}"

                    systs.append(syst_var)
                    wtypes.append(wtype_var)

                if list_of_tuples and systs:
                    syst_list.append(tuple(systs))
                    wtype_list.append(tuple(wtypes))
                else:
                    syst_list += systs
                    wtype_list += wtypes

    if get_weight_types:
        return syst_list, wtype_list
    else:
        return syst_list

# uncertainty groups
uncertainty_groups = {
    "JES" : {
        "label" : "JES/JER",
        "filters" : ["CategoryReduction_JET_", "weight_jvt"],
        "style" : {
            "edgecolor" : "red", "facecolor": "none", "linestyle": "-"
        }
    },
    "BTag" : {
        "label" : "Flavor Tagging",
        "filters" : ["bTagSF_DL1r_"],
        "style" : {
            "edgecolor" : "grey", "facecolor": "none", "linestyle": "-"
        }
    },
    "Lepton" : {
        "label" : "Lepton",
        "filters" : ["EG_", "MUON_", "leptonSF_"],
        "style" : {
            "edgecolor" : "red", "facecolor": "none", "linestyle": ":"
        }
    },
    "MET" : {
        "label" : "$E_{\text{T}}^{\text{miss}}$",
        "filters" : ["MET_"],
        "style" : {
            "edgecolor" : "orange", "facecolor": "none", "linestyle": ":"
        }
    },
    "Backgrounds" : {
        "label" : "Backgrounds",
        "filters" : ['singleTop_', 'VV_', 'ttV_', 'Wjets_', 'Zjets_', 'fakes_'],
        "style" : {
            "edgecolor" : "tab:cyan", "facecolor": "none", "linestyle": "-"
        }
    },
    "Pileup" : {
        "label" : "Pileup",
        "filters" : ["pileup_UP", "pileup_DOWN"],
        "style" : {
            "edgecolor" : "tan", "facecolor": "none", "linestyle": "-"
        }
    },
    "IFSR" : {
        "label" : "IFSR",
        "filters" : ['scale_mu','isr_','fsr_'],
        "style" : {
            "edgecolor" : "tab:purple", "facecolor": "none", "linestyle": "-."
        }
    },
    "PDF" : {
        "label" : "PDF",
        "filters" : ['PDF4LHC15_'],
        "style" : {
            "edgecolor" : "purple", "facecolor": "none", "linestyle": "-."
        },
        "shape_only" : True
    },
    "MTop" : {
        "label" : "$m_{\text{t}}$",
        "filters" : ["mtop_"],
        "style" : {
            "edgecolor" : "yellow", "facecolor": "none", "linestyle": "-"
        }
    },
    "hdamp" : {
        "label" : "$h_{\text{damp}}$ variation",
        "filters" : ["hdamp"],
        "style" : {
            "edgecolor" : "purple", "facecolor": "none", "linestyle": "--"
        }
    },
    "Hadronization" : {
        "label" : "Hadronization",
        "filters" : ["ps_hw"],
        "style" : {
            "edgecolor" : "green", "facecolor": "none", "linestyle": "--"
        }
    },
    "Generator" : {
        "label" : "Hard Scattering",
        "filters" : ['generator_amc'],
        "style" : {
            "edgecolor" : "blue", "facecolor": "none", "linestyle": "--"
        }
    },
    "Lumi" : {
        "label" : "Luminosity",
        "filters" : ['lumi'],
        "style" : {
            "edgecolor" : "yellow", "facecolor": "none", "linestyle": "-."
        }
    },
    "Network" : {
        "label" : "Network",
        "filters" : ['network'],
        "style" : {
            "edgecolor" : "tan", "facecolor": "none", "linestyle": "--"
        }
    },
    "Unfold" : {
        "label" : "Unfold",
        "filters" : ['unfold'],
        "style" : {
            "edgecolor" : "tab:brown", "facecolor": "none", "linestyle": "-."
        }
    },
    "MCStat": {
        "label" : "MC Stat.",
        "style" : {
            "edgecolor" : "darkviolet", "facecolor": "none", "linestyle": "--"
        }
    },
    "stat_total" : {
        "label" : "Stat. Unc.",
        "filters" : ['stat_total'],
        "style" : {
            "edgecolor" : "none", "facecolor": "black", "alpha": 0.25, "zorder" : -1
        }
    },
    "Total" : {
        "label" : "Syst.+Stat. Unc.",
        "filters" : ['total'],
        "style" : {
            "edgecolor" : "none", "facecolor": "black", "alpha": 0.1, "zorder" : -2
        }
    },
    # combined
    "Lepton+MET" : {
        "label": "Lepton, $E_{\text{T}}^{\text{miss}}$",
        "filters" : ["EG_", "MUON_", "leptonSF_", "MET_"],
    },
}