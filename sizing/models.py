from django.db import models
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Create your models here.

class LUT:

    """
    Lookup Table (LUT) class for MOS transistor

    This class loads transistor data from a text file into a Pandas DataFrame
    and allowed access to device parameters for specific operating conditions.
    The LUT contains electrical characteristics such as drain current (Id), 
    transconductance (gm), different capacitances (Cgs, Cgd, Cdd) and voltages (Vgs, Vds).

    Attributes
    ----------
    NCH : pandas.DataFrame
        Lookup table containing transistor characterization data.

    Methods
    -------
    get_lengths()
        Returns all available channel lengths stored in the LUT.

    get_data(target, Vgs=None, Vds=0.6, L=0.13e-6)
        Returns values of the requested parameter. If Vgs is specified, the calue is obtained by linear
        interpolation between the nearest LUT entries. Otherwise, all values corresponding to the selected
        length and Vds are returned.

    Parameters stored in the LUT
    ----------------------------
    Length1 / Length : Transistor length
    cdd : Drain capacitance
    cgd : Gate-Drain capacitance
    cgg : Total gate capacitance
    cgs : Gate-Source capacitance
    gds : Output Conductance
    gm : Transconductance
    Id : Drain current
    Vth : Threshold voltage
    Vds : Drain-Source voltage
    Vgs : Surce-Gate voltage
    """

    def __init__(self, filename):
        self.NCH = pd.read_csv(filename, sep=r"\s+")

        # Spalten auf einfache Namen umbenennen
        self.NCH.columns = ["Length1", "cdd", "cgd", "cgg", "cgs", "gds", "gm", "Id", "Length", "Vth", "Vds", "Vgs"]

    def get_lengths(self):
        return np.sort(self.NCH["Length"].unique())

    def get_data(self, target, Vgs=None, Vds=0.6, L=0.13e-6):

        subset = self.NCH[
            (self.NCH["Length"] == L) &
            (self.NCH["Vds"] == Vds)
        ]

        if Vgs is None:
            return subset[target].to_numpy()

        x = subset["Vgs"].to_numpy()
        y = subset[target].to_numpy()

        idx = np.argsort(x)

        return np.interp(Vgs, x[idx], y[idx])


class DesignSpecification:

    """
    Class for circuit design specifications.

    This class splitts into two distinguisched Design Specifications. First one is Fixed and no negotiable 
    specifications: Vdd, CL and RS. Second one are the Design Choices make from Designer himself
     such as gm/Id, L or RD. And these will be sweeped for making a Desgin Space.
    All of these are distinguisched in the GUI in different Tables and also in the class
    with the suffix "spec". 

    Parameters
    ----------
    vdd : float
        Supply voltage of the circuit on Volts (V)

    CL : float
        Load capacitance connected to the circuit output in farads (F)
        
    RS : float
        Source resistance (eg. Sensor) in Ohms

    gm_id_spec : float
        Target transconductance efficiency (gm/Id), the core parameter 

    L_spec : float
        Desired transistor channel length im micrometers (um)
    
    RD_spec : float
        Drain resistance in Ohms
    """

    def __init__(self, vdd, CL, RS, gm_id_spec, L_spec, RD_spec):
        self.vdd = vdd
        self.CL = CL
        self.RS = RS
        self.gm_id_spec = gm_id_spec
        self.L_spec = L_spec
        self.RD_spec = RD_spec


class CircuitType:

    """
    Base class for all circuit topologies

    This class is common for different circuit implementations.
    It currently does not provide any functionality, but it defines a common
    interface that can be extended in the future.
    """

    def __init__(self):
        pass


class CommonSource(CircuitType):

    """
    This class implement a gm/Id based design methodology for a common-Source
    amplifier. The design flow consists of the following steps:

    Algorithm in 5 steps:
    1. Determine the maximum achivable intrinsic gain (Av0)
    2. Calculate the required transconductance (gm)
    3. Compute drain current (Id) and transistor width (W)
    4. Estimate parasitic capacitances (Cgs, Cgd, Cdd, Cdb)
    5. Calculate the amplifier bandwidth (bw)

    Atributes
    ---------
    spec : DesignSpecification
        This attribut allowed access to Design Specifications.
    
    results : DataFrame
        This Data Frame containing Width independent Parameters such as gm/Id, gm/gds and so on.
        This Data Frame is produced by Calculator class defined below.

    Main Outputs
    ------------
    Av_max : Maximum Low Frequency Voltage Gain

    gm : Required transistor transconductance

    Id : Required drain current

    W : Required transistor width

    Cgs, Cgd, Cdd, Cdb : Estimated transistor parasitic capacitances

    bw : estimated -3dB bandwidth of the amplifier

    Notes
    -----
    Most electrical quantities are obtained through linear interpolation for 
    a desired Transistor length (L) and the Drain-Source voltage (Vds).
    This class at the end performes parameter sweep over gm/Id and RD to explore the 
    design space. 
    """

    def __init__(self, spec, results):
        super().__init__()
        self.spec = spec
        self.results = results

############ STEP 1 FIND Av0_max ####################

    def calculate_avmax(self, gm_id):

        data = self.results[self.results["Length"] == self.spec.L_spec]

        av_max = -np.inf
        best_vds = None
        rows = []

        for vds, subset in data.groupby("Vds"):

            vea = self.calculate_vea(gm_id, subset)

            try:
                av = gm_id / (1 / vea + 1 / (1.21 - vds))
            except Exception:
                print(f"vea={vea}, vds={vds}")

            rows.append((vds, av))
            #print(f"vds: {vds}, av: {av}")

            if av > av_max:
                av_max = av
                best_vds = vds

        return pd.DataFrame(rows, columns=["Vds", "Av"]), av_max, best_vds
    
    
    def calculate_vea(self, gm_id, subset):

        x = subset["gm_id_lut"].to_numpy()
        y = subset["id_gds_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])

    
############# STEP 2 FIND gm ###############

    def calculate_gm(self, gm_id, vds):
        
        _, av_max, best_vds = self.calculate_avmax(gm_id)

        #return 10e-3
        return (1/self.spec.RD_spec) / ( (1/av_max) - (1/self.calculate_gm_gds(gm_id, vds)) )
    

    def calculate_gm_gds(self, gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["gm_gds_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    
############# STEP 3 FIND Id and W ###################

    def calculate_Id(self, gm_id, vds):
        return self.calculate_gm(gm_id, vds)/gm_id
    
    def calculate_W(self,gm_id, vds):
        Id = self.calculate_Id(gm_id, vds)
        id_w = self.calculate_id_w(gm_id, vds)

        return Id/id_w
    
    def calculate_id_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["id_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])

########### STEP 4 FIND Cgs Cgd Cdd #################

    def calculate_cgs_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cgs_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cgs(self, gm_id, vds):
        return self.calculate_W(gm_id, vds) * self.calculate_cgs_w(gm_id, vds)
    
    def calculate_cgd_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cgd_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cgd(self, gm_id, vds):
        return self.calculate_W(gm_id, vds) * self.calculate_cgd_w(gm_id, vds)
    
    def calculate_cdd_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cdd_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cdd(self, gm_id, vds):
        return (self.calculate_W(gm_id, vds) * self.calculate_cdd_w(gm_id, vds))
    
    def calculate_cdb(self, gm_id, vds):
        return (self.calculate_cdd(gm_id, vds) - self.calculate_cgd(gm_id, vds))
    
    def calculate_CLtotal(self, gm_id, vds):
        return (self.spec.CL + self.calculate_cdb(gm_id, vds))
    
################ STEP 5 FIND bandwidth ###################

    def calculate_bw(self, gm_id, vds):
        _, av_max, best_vds = self.calculate_avmax(gm_id)

        tau = ( self.spec.RS*(self.calculate_cgs(gm_id, vds) + self.calculate_cgd(gm_id, vds)*(1+av_max)) + 
                self.calculate_RDtotal(gm_id, vds)*(self.calculate_CLtotal(gm_id, vds)+ self.calculate_cgd(gm_id, vds)) )

        return 1/2/np.pi/tau
    
    def calculate_rds(self, gm_id, vds):
        return self.calculate_gm_gds(gm_id, vds) / self.calculate_gm(gm_id, vds)
    
    def calculate_RDtotal(self, gm_id, vds):
        return (self.calculate_rds(gm_id, vds)*self.spec.RD_spec)/(self.calculate_rds(gm_id, vds)+self.spec.RD_spec)

###########################################################

    
    def calculate_ft(self, gm_id, vds):             # NOT IN USE

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["ft_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    

################ SWEEP ####################################
    
    def size_sweep(self, gmid_min=5, gmid_max=25, gmid_step=1, RD_min=100, RD_max=1000, RD_step=100):
        rows = []

        L = self.spec.L_spec

        for RD in  np.arange(RD_min, RD_max+RD_step, RD_step):
            self.spec.RD_spec = RD

            for gmid in np.arange(gmid_min, gmid_max+gmid_step, gmid_step):

                #best_vds = 0.6
                _, av_max, best_vds = self.calculate_avmax(gmid)
                
                gm = self.calculate_gm(gmid, best_vds)
                gm_gds = self.calculate_gm_gds(gmid, best_vds)

                Id = self.calculate_Id(gmid, best_vds)
                W = self.calculate_W(gmid, best_vds)
                id_w = self.calculate_id_w(gmid, best_vds)
                
                cgs = self.calculate_cgs(gmid, best_vds)
                cgd = self.calculate_cgd(gmid, best_vds)
                cdd = self.calculate_cdd(gmid, best_vds)
                cdb = self.calculate_cdb(gmid, best_vds)
                CLtotal = self.calculate_CLtotal(gmid, best_vds)

                rds = self.calculate_rds(gmid, best_vds)
                RDtotal = self.calculate_RDtotal(gmid, best_vds)
                RD = self.spec.RD_spec
                bw = self.calculate_bw(gmid, best_vds)
                av0 = av_max
                

                rows.append({
                    "gm_id": gmid,
                    "Vdsmax": best_vds,
                    "gm": gm,
                    "Id": Id,
                    "Id/W": id_w,
                    "W": W,
                    "L": L,
                    "cgs": cgs,
                    "cgd": cgd,
                    "cdd": cdd,
                    "cdb": cdb,
                    "CLtotal": CLtotal,
                    "rds": rds,
                    "RDtotal": RDtotal,
                    "RD": RD,
                    "bw": bw,
                    "av0": av0,
                })
            
        return pd.DataFrame(rows)

class Vectors:
    """
    This class provides acess to transistor parameters stored in the Lookup Table
     (LUT). The extracted vectors are Width dependent. 

    This call will be used for Sanity Check of the Data. For generating a Plot Id(Vgs).

    Attributes
    ----------
    lut : Lookup Table (already described)

    spec : DesignSpecification (already described)

    Most Important Methods
    ----------------------
    id_lut(Vgs)
        Returns the drain current corresponding to a given Vgs
    """
    
    def __init__(self, lut, spec):
        self.lut = lut
        self.spec = spec

    def vgs_lut(self):
        return self.lut.get_data("Vgs" , Vds=0.6, L=self.spec.L_spec)

    def id_lut(self, Vgs):
        return self.lut.get_data("Id" , Vgs=Vgs,  Vds=0.6, L=self.spec.L_spec)
    
    def gm_lut(self, Vgs):
        return self.lut.get_data("gm" , Vds=0.6, L=self.spec.L_spec)
    
    def gds_lut(self, Vgs):
        return self.lut.get_data("gds" , Vds=0.6, L=self.spec.L_spec)
    
    def cgg_lut(self, Vgs):
        return self.lut.get_data("cgg" , Vds=0.6, L=self.spec.L_spec)
    
    def cgs_lut(self, Vgs):
        return self.lut.get_data("cgs" , Vds=0.6, L=self.spec.L_spec)
    
    def cgd_lut(self, Vgs):
        return self.lut.get_data("cgd" , Vds=0.6, L=self.spec.L_spec)
    

class Equations:
    """
    Collection of transistor equations used in gm/Id design methodology.
    The resulting quantitis are independent of transistor width.
    This class is a Help class for the following class Calculator.

    Most Important Methods
    ----------------------
    gm_id_lut(gm, ids)
        Calculate the transistor efficiency (gm/Id)

    """

    def gm_id_lut(self, gm, ids):
        return gm / ids

    def gm_gds_lut(self, gm, gds):
        return gm / gds

    def ft_lut(self, gm, cgg):
        return gm/cgg/2/np.pi

    def id_w_lut(self, ids, w):
        return ids/w

    def id_gds_lut(self, ids, gds):         # Early Voltage Vea=Id/gds
        return ids/gds
    
    def cgs_w_lut(self, cgs, w):
        return cgs/w
    
    def cgd_w_lut(self, cgd, w):
        return cgd/w
    
    def cdd_w_lut(self, cdd, w):
        return cdd/w


class Calculator:
    """
    This class processes the data stored in a LUT and generates a new DataFrame.
    The previous class Equations ist used for it.

    Attributes
    ----------
    lut : pd.DataFrame
        Main Pandas DataFrame given as an argument  

    width : float
        Transistor width is known w=5e-6

    DataFrame Columns
    -----------------
    Length | Vgs | Vds | gm_id_lut | gm_gds_lut | ft_lut | id_w_lut | id_gds_lut | cgs_w_lut | cgd_w_lut | cdd_w_lut

    """

    def __init__(self, lut, width):
        self.lut = lut
        self.width = width

    def calculate(self):

        NCH = self.lut.NCH
        equations = Equations()
        result = pd.DataFrame()

        result["Length"] = NCH["Length"]
        result["Vgs"] = NCH["Vgs"]
        result["Vds"] = NCH["Vds"]

        result["gm_id_lut"] = equations.gm_id_lut(
            NCH["gm"],
            NCH["Id"]
        )

        result["gm_gds_lut"] = equations.gm_gds_lut(
            NCH["gm"],
            NCH["gds"]
        )

        result["ft_lut"] = equations.ft_lut(
            NCH["gm"],
            NCH["cgg"]
        )

        result["id_w_lut"] = equations.id_w_lut(
            NCH["Id"],
            self.width
        )

        result["id_gds_lut"] = equations.id_gds_lut(
            NCH["Id"],
            NCH["gds"]
        )

        result["cgs_w_lut"] = equations.cgs_w_lut(
            NCH["cgs"],
            self.width
        )

        result["cgd_w_lut"] = equations.cgd_w_lut(
            NCH["cgd"],
            self.width
        )

        result["cdd_w_lut"] = equations.cdd_w_lut(
            NCH["cdd"],
            self.width
        )

        return result


class Plot:
    """
    This class provides a collection of Plotly based plotting functions used to
    visualise Design Space and typical gm/Id curves.
    All plots are returned as HTML objects and can be directly embedded into a 
    web application.

    Methods
    -------
        Design Space
        ------------
            plot_bw_id(df)
            plot_avmax_vds(df_avmax)

        Typical gm/Id Curves
        --------------------
            plot_idw_gmid(results, lut)
            plot_gmgds_gmid(results, lut)
            plot_ft_gmid(results, lut)
            plot_idgds_gmid(results, lut)
            plot_gmid_ft_idw(results, lut)

        Plausibility Check
        ------------------
            plot_id_vgs(Vgs, Id)

    """

    def __init__(self):
        pass

############## Design Space #################

    def plot_bw_id(self, df):

        fig = go.Figure()

        for L, subset in df.groupby("L"):

            fig.add_trace(
                go.Scatter(
                    x=subset["Id"],
                    y=subset["bw"],
                    mode="markers",
                    name=f"L = {L*1e9:.0f} nm",

                    customdata=np.column_stack((
                        subset["gm_id"],
                        subset["L"] * 1e6,      # um
                        subset["W"] * 1e6,      # µm
                        subset["RD"],
                        subset["Id"] * 1e6,     # µA
                        subset["av0"],          # V/V
                        subset["bw"] * 1e-6,    # MHz
                        
                    )),

                    hovertemplate=
                        "gm/Id = %{customdata[0]:.1f}<br>"
                        "L = %{customdata[1]:.2f} µm<br>"
                        "W = %{customdata[2]:.2f} µm<br>"
                        "RL = %{customdata[3]:.2f} Ω<br>"
                        "Id = %{customdata[4]:.2f} µA<br>"
                        "Av0 = %{customdata[5]:.2f} V/V<br>"
                        "BW = %{customdata[6]:.3e} MHz<br>"
                        "<extra></extra>"
                )
            )

        fig.update_layout(
            title="Bandwidth",
            xaxis_title="Id (A)",
            yaxis_title="BW (Hz)",
            template="plotly_white"
        )

        return fig.to_html()
    
############## Maximum DC Voltage Gain ####################

    def plot_avmax_vds(self, df_avmax):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_avmax["Vds"],
                y=df_avmax["Av"],
                mode="lines+markers",
                name="Av max",
                customdata=np.column_stack((
                    df_avmax["Vds"],
                    df_avmax["Av"]
                )),

                hovertemplate=
                    "Vds: %{x:.2f}<br>"
                    "Avmax: %{y:.2f}<br>"
                    "<extra></extra>"
            )
        )

        fig.update_layout(
            height=450,
            title="Maximum Voltage Gain",
            xaxis_title="Vds (V)",
            yaxis_title="Av max (V/V)",
            template="plotly_white"
        )

        return fig.to_html()


################### Typical gm/Id Plots ###################

    def plot_idw_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == 0.6)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["id_w_lut"],
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(width=2),
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                        "gm/Id = %{x:.2f}<br>" +
                        "Id/W = %{y:.2e}<br>" +
                        "<extra></extra>"
                )
            )

        fig.update_layout(
            height=450,
            title="Current Density",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="Id/W (A/m)",
            yaxis=dict(type="log",
                       range=[-1,None],         # Y-axis starts by 10^-1
                       ),
            template="plotly_white"
        )

        return fig.to_html()
    
    
    def plot_gmgds_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == 0.4)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["gm_gds_lut"],
                    mode="lines+markers",
                    customdata=np.column_stack((
                        np.full(len(data["gm_id_lut"]), length),
                        data["id_w_lut"]
                    )),
                    name=f"length = {length*1e9:.0f} nm"
                )
            )

        fig.update_layout(
            height=450,
            title="Intrinsic Gain",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="gm/gds (V/V)",
            template="plotly_white"
        )

        return fig.to_html()
    
    def plot_ft_gmid(self, results, lut):
        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == 0.4)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["ft_lut"],
                    mode="lines+markers",
                    marker=dict(size=6),
                    line=dict(width=2),
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                        "gm/Id = %{x:.2f}<br>" +
                        "fT = %{y:.2e}<br>" +
                        "<extra></extra>"
                )
            )

        fig.update_layout(
            height=450,
            title="Transit Frequency",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="fT (Hz)",
            template="plotly_white"
        )

        return fig.to_html()


    def plot_idgds_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == 0.4)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["id_gds_lut"],
                    mode="lines+markers",
                    customdata=np.column_stack((
                        np.full(len(data["gm_id_lut"]), length),
                        data["id_w_lut"]
                    )),
                    name=f"length = {length*1e9:.0f} nm"
                )
            )

        fig.update_layout(
            height=450,
            title="Early Voltage",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="Vea (V)",
            template="plotly_white"
        )

        return fig.to_html()

    
    def plot_gmid_ft_idw(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == 0.4)]

            # Left y-axis: gm/Id
            fig.add_trace(
                go.Scatter(
                    x=data["id_w_lut"],
                    y=data["gm_id_lut"],
                    mode="lines+markers",
                    name=f"gm/Id ({length*1e9:.0f} nm)",
                    marker=dict(size=6),
                    line=dict(width=2),
                    yaxis="y"
                )
            )

            # Right y-axis: fT
            fig.add_trace(
                go.Scatter(
                    x=data["id_w_lut"],
                    y=data["ft_lut"],
                    mode="lines+markers",
                    name=f"fT ({length*1e9:.0f} nm)",
                    marker=dict(size=6),
                    line=dict(width=2, dash="dash"),
                    yaxis="y2"
                )
            )

        fig.update_layout(

            height=500,

            title="gm/Id and fT vs Id/W",

            xaxis=dict(
                title="Id/W (A/m)",
                type="log"
            ),

            yaxis=dict(
                title="gm/Id (S/A)"
            ),

            yaxis2=dict(
                title="fT (Hz)",
                overlaying="y",
                side="right",
                type="log"
            ),

            template="plotly_white",

            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig.to_html()
    
############# Plausibility check #####################

    def plot_id_vgs(self, Vgs, Id):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=Vgs, 
                y=Id, 
                mode='lines+markers', 
                name=f'Id vs Vgs)'
            )
        )

        fig.update_layout(
            height=450,
            title='Id vs Vgs', 
            xaxis_title='Vgs (V)', 
            yaxis_title='Id (A)',
            template="plotly_white"
        )
    
        return fig.to_html()
    
############### Call the Functions ###############

# einmalig laden
lut = LUT("data/nmos.txt")
calculator = Calculator(lut=lut, width=5e-6)
RESULTS = calculator.calculate()

plotter = Plot()
GRAPH2 = plotter.plot_idw_gmid(RESULTS, lut)
GRAPH3 = plotter.plot_gmgds_gmid(RESULTS, lut)
GRAPH4 = plotter.plot_ft_gmid(RESULTS, lut)
GRAPH5 = plotter.plot_idgds_gmid(RESULTS, lut)

############### END #################################
