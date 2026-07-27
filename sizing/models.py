from django.db import models
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DEFAULT_VDS = 0.6
DEFAULT_LENGTH = 0.13e-6
DEFAULT_WIDTH = 5e-6

# Create your models here.

class LUT:

    """
    Lookup table (LUT) for MOS transistor characterization data.

    Loads transistor operating-point data from a text file into a Pandas
    DataFrame and provides access to electrical parameters such as Id, gm,
    gds, capacitances, and terminal voltages through filtering and
    interpolation.

    Attributes
    ----------
    NCH : pandas.DataFrame

    """

    def __init__(self, filename):
        self.NCH = pd.read_csv(filename, sep=r"\s+")

        # Spalten auf einfache Namen umbenennen
        self.NCH.columns = ["Length1", "cdd", "cgd", "cgg", "cgs", "gds", "gm", "Id", "Length", "Vth", "Vds", "Vgs"]

    def get_lengths(self):
        return np.sort(self.NCH["Length"].unique())

    def get_data(self, target, Vgs=None, Vds=DEFAULT_VDS, L=DEFAULT_LENGTH):

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
    Fixed specifications are Vdd, CL, and RS. Design choices are gm/Id,
    transistor length, and drain resistance, which can be varied during
    design-space exploration.

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
    Base class for all circuit topologies.

    Defines a common parent type for circuit implementations.
    """

    def __init__(self):
        pass


class CommonSource(CircuitType):

    """
    Implements a gm/Id-based design methodology for a common-source amplifier.

    The design procedure consists of:
    1. Determining the maximum achievable intrinsic gain.
    2. Calculating the required transconductance.
    3. Computing drain current and transistor width.
    4. Estimating parasitic capacitances.
    5. Calculating the amplifier bandwidth.

    Attributes
    ----------
    spec : DesignSpecification
        Circuit design specifications and design choices.
    results : pandas.DataFrame
        Width-independent design parameters generated during the sizing process.

    Notes
    -----
    Device parameters are obtained from lookup tables using linear
    interpolation for the selected channel length and drain-source voltage.
    The implementation performs parameter sweeps over gm/Id and drain
    resistance to explore the design space.

    """

    def __init__(self, spec, results):
        super().__init__()
        self.spec = spec
        self.results = results

#################### STEP 1 FIND Av0_max ####################
    
    #@profile
    def calculate_avmax(self, gm_id):

        data = self.results[self.results["Length"] == self.spec.L_spec]

        av_max = -np.inf
        best_vds = None
        rows = []

        for vds, subset in data.groupby("Vds"):

            vea = self.calculate_vea(gm_id, subset)
            av = gm_id / ( (1/vea) + (1/(self.spec.vdd - vds)) )

            rows.append((vds, av))

            if av > av_max:
                av_max = av
                best_vds = vds

        return pd.DataFrame(rows, columns=["Vds", "Av"]), av_max, best_vds
    
    #@profile
    def calculate_vea(self, gm_id, subset):

        x = subset["gm_id_lut"].to_numpy()
        y = subset["id_gds_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])

    
#################### STEP 2 FIND gm ####################

    def calculate_gm(self, gm_id, vds, av):

        #return 10e-3
        return (1/self.spec.RD_spec) / ( (1/av) - (1/self.calculate_gm_gds(gm_id, vds)) )
    

    def calculate_gm_gds(self, gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        #print(data.to_string())

        x = data["gm_id_lut"].to_numpy()
        y = data["gm_gds_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    
#################### STEP 3 FIND Id and W ####################

    def calculate_Id(self, gm_id, vds, av_max):
        return self.calculate_gm(gm_id, vds, av_max)/gm_id
    
    def calculate_W(self,gm_id, vds, av_max):
        Id = self.calculate_Id(gm_id, vds, av_max)
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

#################### STEP 4 FIND Cgs Cgd Cdd ####################

    def calculate_cgs_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cgs_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cgs(self, gm_id, vds, av_max):
        return self.calculate_W(gm_id, vds, av_max) * self.calculate_cgs_w(gm_id, vds)
    
    def calculate_cgd_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cgd_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cgd(self, gm_id, vds, av_max):
        return self.calculate_W(gm_id, vds, av_max) * self.calculate_cgd_w(gm_id, vds)
    
    def calculate_cdd_w(self,gm_id, vds):

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["cdd_w_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    
    def calculate_cdd(self, gm_id, vds, av_max):
        return (self.calculate_W(gm_id, vds, av_max) * self.calculate_cdd_w(gm_id, vds))
    
    def calculate_cdb(self, gm_id, vds, av_max):
        return (self.calculate_cdd(gm_id, vds, av_max) - self.calculate_cgd(gm_id, vds, av_max))
    
    def calculate_Cout(self, gm_id, vds, av_max):
        return (self.spec.CL + self.calculate_cdb(gm_id, vds, av_max))
    
#################### STEP 5 FIND bandwidth ####################

    def calculate_bw(self, gm_id, vds, av_max):
        #_, av_max, best_vds = self.calculate_avmax(gm_id)
        cgs = self.calculate_cgs(gm_id, vds, av_max)
        cgd = self.calculate_cgd(gm_id, vds, av_max)
        rout = self.calculate_Rout(gm_id, vds, av_max)
        cout = self.calculate_Cout(gm_id, vds, av_max)

        tau = ( self.spec.RS*(cgs + cgd*(1+av_max)) + rout*(cout+cgd) )

        return 1/2/np.pi/tau
    
    def calculate_rds(self, gm_id, vds, av_max):
        return self.calculate_gm_gds(gm_id, vds) / self.calculate_gm(gm_id, vds, av_max)
    
    def calculate_Rout(self, gm_id, vds, av_max):
        rds = self.calculate_rds(gm_id, vds, av_max)

        return (rds*self.spec.RD_spec) / (rds+self.spec.RD_spec)

###############################################################

    
    def calculate_ft(self, gm_id, vds):             # NOT IN USE

        data = self.results[
            (self.results["Length"] == self.spec.L_spec) & 
            (self.results["Vds"] == vds)
        ]

        x = data["gm_id_lut"].to_numpy()
        y = data["ft_lut"].to_numpy()

        idx = np.argsort(x)

        return np.interp(gm_id, x[idx], y[idx])
    

#################### SIZE SWEEP ###############################
    
    def size_sweep(self, gmid_min=5, gmid_max=25, gmid_step=5, RD_min=1000, RD_max=1000, RD_step=1000):
        rows = []

        L = self.spec.L_spec

        for RD in  np.arange(RD_min, RD_max+RD_step, RD_step):
            self.spec.RD_spec = RD      # be aware this assignment change the spec value!

            for gmid in np.arange(gmid_min, gmid_max+gmid_step, gmid_step):

                #best_vds = DEFAULT_VDS
                _, av_max, best_vds = self.calculate_avmax(gmid)
                
                gm = self.calculate_gm(gmid, best_vds, av_max)

                Id = self.calculate_Id(gmid, best_vds, av_max)
                W = self.calculate_W(gmid, best_vds, av_max)
                id_w = self.calculate_id_w(gmid, best_vds)
                
                cgs = self.calculate_cgs(gmid, best_vds, av_max)
                cgd = self.calculate_cgd(gmid, best_vds, av_max)
                cdd = self.calculate_cdd(gmid, best_vds, av_max)
                cdb = self.calculate_cdb(gmid, best_vds, av_max)
                Cout = self.calculate_Cout(gmid, best_vds, av_max)

                rds = self.calculate_rds(gmid, best_vds, av_max)
                Rout = self.calculate_Rout(gmid, best_vds, av_max)
                RD = self.spec.RD_spec
                bw = self.calculate_bw(gmid, best_vds, av_max)
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
                    "Cout": Cout,
                    "rds": rds,
                    "Rout": Rout,
                    "RD": RD,
                    "bw": bw,
                    "av0": av0,
                })
            
        return pd.DataFrame(rows)

class Vectors:

    """
    Accesses and visualizes width-dependent transistor parameters from the LUT.

    Primarily used for LUT validation and plotting transistor
    characteristics such as Id(Vgs).

    """
    
    def __init__(self, lut, spec):
        self.lut = lut
        self.spec = spec

    def vgs_lut(self):
        return self.lut.get_data("Vgs" , Vds=DEFAULT_VDS, L=self.spec.L_spec)

    def id_lut(self, Vgs):
        return self.lut.get_data("Id" ,  Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def gm_lut(self, Vgs):
        return self.lut.get_data("gm" ,  Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def gds_lut(self, Vgs):
        return self.lut.get_data("gds" , Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def cgg_lut(self, Vgs):
        return self.lut.get_data("cgg" , Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def cgs_lut(self, Vgs):
        return self.lut.get_data("cgs" , Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def cgd_lut(self, Vgs):
        return self.lut.get_data("cgd" , Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    
    def cdd_lut(self, Vgs):
        return self.lut.get_data("cdd" , Vgs=Vgs, Vds=DEFAULT_VDS, L=self.spec.L_spec)
    

class Equations:

    """
    Provides width-independent transistor calculations and serves as a
    helper class for the Calculator implementation.

    Methods
    -------
    gm_id_lut(gm, ids)
        Returns the transconductance efficiency (gm/Id).

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
    Processes LUT data and generates a derived DataFrame containing
    gm/Id design parameters.

    The calculations are based on the transistor equations provided by the
    Equations class.

    Attributes
    ----------
    lut : pandas.DataFrame
        Input lookup table containing transistor characterization data.
    width : float
        Reference transistor width used for normalization.

    Generated Columns
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
    Collection of Plotly-based visualization functions.

    Supports:
    - Design-space exploration plots
    - gm/Id design curves
    - LUT plausibility checks

    All plots are returned as HTML objects for direct integration into a
    web application.
    """

    def __init__(self):
        pass

#################### Design Space ####################

    def plot_bw_id(self, df):

        fig = go.Figure()

        for L, subset in df.groupby("L"):

            fig.add_trace(
                go.Scatter(
                    x=subset["Id"],
                    y=subset["bw"],
                    mode="markers",
                    marker=dict(size=12),
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
                        "BW = %{customdata[6]:.2f} MHz<br>"
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
    
#################### Maximum DC Voltage Gain ####################

    def plot_avmax_vds(self, df_avmax):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df_avmax["Vds"],
                y=df_avmax["Av"],
                mode="lines+markers",
                name="Av max",
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


#################### Typical gm/Id Plots ####################

    def plot_idw_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == DEFAULT_VDS)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["id_w_lut"],
                    mode="lines+markers",
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "Id/W: %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )

        fig.update_layout(
            height=450,
            title="Current Density",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="Id/W (A/m)",
            yaxis=dict(type="log", range=[-1,None]),        # Y-axis starts by 10^-1
            template="plotly_white"
        )

        return fig.to_html()
    
    
    def plot_gmgds_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == DEFAULT_VDS)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["gm_gds_lut"],
                    mode="lines+markers",
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "gm/gds: %{y:.2f}<br>"
                    "<extra></extra>"
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

            data = results[(results["Length"] == length) & (results["Vds"] == DEFAULT_VDS)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["ft_lut"] / 1e9,         # store value in GHz
                    mode="lines+markers",
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "fT: %{y:.2f} GHz<br>"
                    "<extra></extra>"
                )
            )

        fig.update_layout(
            height=450,
            title="Transit Frequency",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="fT (GHz)",
            template="plotly_white"
        )

        return fig.to_html()


    def plot_idgds_gmid(self, results, lut):

        fig = go.Figure()

        for length in lut.get_lengths():

            data = results[(results["Length"] == length) & (results["Vds"] == DEFAULT_VDS)]

            fig.add_trace(
                go.Scatter(
                    x=data["gm_id_lut"],
                    y=data["id_gds_lut"],
                    mode="lines+markers",
                    name=f"length = {length*1e9:.0f} nm",
                    hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "Vea: %{y:.2f}<br>"
                    "<extra></extra>"
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
    
    def plot_capacitances_gmid(self, results):

        fig = go.Figure()

        data = results[(results["Length"] == DEFAULT_LENGTH) & (results["Vds"] == DEFAULT_VDS)]

        fig.add_trace(
            go.Scatter(
                x=data["gm_id_lut"],
                y=data["cgs_w_lut"] * 1e12,     # store value in pF
                mode="lines+markers",
                name="Cgs/W",
                hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "Cgs/W: %{y:.2f} pF<br>"
                    "<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["gm_id_lut"],
                y=data["cgd_w_lut"] * 1e12,     # store value in pF
                mode="lines+markers",
                name="Cgd/W",
                hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "Cgd/W: %{y:.2f} pF<br>"
                    "<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data["gm_id_lut"],
                y=data["cdd_w_lut"] * 1e12,     # store value in pF
                mode="lines+markers",
                name="Cdd/W",
                hovertemplate=
                    "gm/Id: %{x:.2f}<br>"
                    "Cgd/W: %{y:.2f} pF<br>"
                    "<extra></extra>"
            )
        )

        fig.update_layout(
            height=450,
            title="Parasitic Capacitances",
            xaxis_title="gm/Id (S/A)",
            yaxis_title="Cxx / W (pF/m)",
            template="plotly_white"
        )

        return fig.to_html()
    
#################### Plausibility check ####################

    def plot_id_vgs(self, Vgs, Id):

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=Vgs, 
                y=Id, 
                mode='lines+markers'
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
    
#################### Call the Functions ####################

# einmalig laden
lut = LUT("data/nmos.txt")
calculator = Calculator(lut=lut, width=DEFAULT_WIDTH)
RESULTS = calculator.calculate()

plotter = Plot()
GRAPH2 = plotter.plot_idw_gmid(RESULTS, lut)
GRAPH3 = plotter.plot_gmgds_gmid(RESULTS, lut)
GRAPH4 = plotter.plot_ft_gmid(RESULTS, lut)
GRAPH5 = plotter.plot_idgds_gmid(RESULTS, lut)
GRAPH6 = plotter.plot_capacitances_gmid(RESULTS)

#################### END ###################################
